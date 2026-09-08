"""Cleaning, label harmonisation and the fitted preprocessing pipeline.

Two distinct responsibilities live here and it is worth keeping them separate
in your head:

1. **Dataset-level cleaning** (:func:`clean_dataset`) - deduplication, label
   mapping, dropping unusable classes. These are *offline* decisions that shape
   the training corpus and must never run at inference time.
2. **Row-level transformation** (:func:`build_preprocessor`) - imputation,
   scaling, encoding. This is a fitted scikit-learn object that is serialised
   with the model and replayed identically on every prediction.

Confusing the two is the classic source of train/serve skew: if you deduplicate
or rebalance at inference you change the distribution the model sees.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, RobustScaler

from src.config import (
    BENIGN_LABEL,
    CATEGORICAL_FEATURES,
    CORE_FEATURES,
    LABEL_MAP,
    LEAKAGE_COLUMNS,
    TRAINING,
    get_logger,
)
from src.data_loader import coerce_numeric, find_label_column
from src.feature_engineering import DERIVED_FEATURES, add_derived_features

logger = get_logger(__name__)

_LABEL_CLEAN = re.compile(r"[^a-z0-9]+")


@dataclass
class CleaningReport:
    """Audit trail of what :func:`clean_dataset` removed and why."""

    rows_in: int = 0
    rows_out: int = 0
    duplicates_removed: int = 0
    null_label_rows_removed: int = 0
    infinite_values_replaced: int = 0
    all_nan_columns_dropped: list[str] = None  # type: ignore[assignment]
    constant_columns_dropped: list[str] = None  # type: ignore[assignment]
    leakage_columns_dropped: list[str] = None  # type: ignore[assignment]
    rare_classes_dropped: dict[str, int] = None  # type: ignore[assignment]
    class_counts: dict[str, int] = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        self.all_nan_columns_dropped = self.all_nan_columns_dropped or []
        self.constant_columns_dropped = self.constant_columns_dropped or []
        self.leakage_columns_dropped = self.leakage_columns_dropped or []
        self.rare_classes_dropped = self.rare_classes_dropped or {}
        self.class_counts = self.class_counts or {}

    def summary(self) -> str:
        return (
            f"{self.rows_in:,} -> {self.rows_out:,} rows | "
            f"-{self.duplicates_removed:,} dupes | "
            f"-{self.null_label_rows_removed:,} null labels | "
            f"{len(self.class_counts)} classes"
        )


def normalize_label(raw: object) -> str:
    """Map a raw CIC-IDS2017 label onto the project's coarse taxonomy.

    Raw labels contain inconsistent separators (``Web Attack \\x96 XSS`` uses a
    Windows-1252 en-dash) so we strip to lowercase alphanumerics before lookup.
    Unrecognised non-benign labels fall back to ``"Other"`` rather than being
    silently discarded - an unexpected label is a data problem worth surfacing.
    """
    text = _LABEL_CLEAN.sub(" ", str(raw).strip().lower()).strip()
    if not text or text == "nan":
        return ""
    if text in LABEL_MAP:
        return LABEL_MAP[text]
    for key, value in LABEL_MAP.items():
        if text.startswith(key) or key in text:
            return value
    return "Other"


def clean_dataset(
    frame: pd.DataFrame,
    label_column: str | None = None,
    min_samples_per_class: int | None = None,
) -> tuple[pd.DataFrame, CleaningReport]:
    """Produce a modelling-ready frame from raw concatenated captures.

    Steps, in order (order matters - we deduplicate *after* dropping identifier
    columns, otherwise near-identical flows separated only by a timestamp would
    survive and inflate the effective dataset size):

    1. Normalise the label column onto the coarse taxonomy.
    2. Drop rows with an unusable label.
    3. Drop identifier / leakage columns.
    4. Replace ``±inf`` with NaN (CICFlowMeter emits ``Infinity`` for
       zero-duration flows in its rate columns).
    5. Drop all-NaN and zero-variance columns - a constant column carries no
       information and only adds noise to scaling and SHAP.
    6. Drop exact duplicate rows.
    7. Drop classes below the viability threshold.

    Returns:
        The cleaned frame (label column renamed to ``label``) and a report.
    """
    report = CleaningReport(rows_in=len(frame))
    min_samples = min_samples_per_class if min_samples_per_class is not None else TRAINING.min_samples_per_class

    label_col = label_column or find_label_column(frame)
    work = frame.copy()
    work["label"] = work[label_col].map(normalize_label)
    if label_col != "label":
        work = work.drop(columns=[label_col])

    before = len(work)
    work = work[work["label"].astype(bool)]
    report.null_label_rows_removed = before - len(work)

    dropped_leak = [c for c in work.columns if c in LEAKAGE_COLUMNS or c == "source_file"]
    work = work.drop(columns=dropped_leak, errors="ignore")
    report.leakage_columns_dropped = dropped_leak

    # Count infinities BEFORE coercion. ``coerce_numeric`` replaces +/-inf with
    # NaN as part of its contract, so counting afterwards would always report
    # zero and this audit field would silently lie about what was cleaned.
    pre_coercion = work.drop(columns=["label"], errors="ignore").apply(
        pd.to_numeric, errors="coerce"
    )
    report.infinite_values_replaced = int(
        np.isinf(pre_coercion.to_numpy(dtype="float64", na_value=np.nan)).sum()
    )

    work = coerce_numeric(work, exclude={"label", "protocol"})
    numeric_cols = work.select_dtypes(include=[np.number]).columns
    work[numeric_cols] = work[numeric_cols].replace([np.inf, -np.inf], np.nan)

    all_nan = [c for c in work.columns if c != "label" and work[c].isna().all()]
    work = work.drop(columns=all_nan)
    report.all_nan_columns_dropped = all_nan

    constant = [
        c
        for c in work.select_dtypes(include=[np.number]).columns
        if work[c].nunique(dropna=True) <= 1
    ]
    work = work.drop(columns=constant)
    report.constant_columns_dropped = constant

    before = len(work)
    work = work.drop_duplicates(ignore_index=True)
    report.duplicates_removed = before - len(work)

    counts = work["label"].value_counts()
    rare = counts[counts < min_samples]
    if not rare.empty:
        report.rare_classes_dropped = rare.to_dict()
        work = work[~work["label"].isin(rare.index)].reset_index(drop=True)
        logger.warning(
            "Dropped %d class(es) with fewer than %d samples: %s",
            len(rare),
            min_samples,
            dict(rare),
        )

    report.rows_out = len(work)
    report.class_counts = work["label"].value_counts().to_dict()
    logger.info("Cleaning complete: %s", report.summary())
    return work, report


def balance_classes(
    frame: pd.DataFrame,
    label_column: str = "label",
    max_per_class: int | None = None,
    random_state: int = TRAINING.random_state,
) -> pd.DataFrame:
    """Cap over-represented classes by random downsampling.

    CIC-IDS2017 is ~80% BENIGN with some attack classes under 2,000 rows - a
    ratio above 1000:1. We handle this in two complementary places:

    * **Here**, by capping the majority class. Downsampling is preferred over
      SMOTE-style oversampling for network data because synthetic flows are not
      physically realisable - interpolating between two TCP flows can produce a
      record with 3.5 SYN flags, which teaches the model a decision boundary
      that no real packet can occupy.
    * **In the model**, via ``class_weight='balanced'`` / ``sample_weight``,
      which raises the loss contribution of the remaining minority rows.

    Only the training split is ever balanced. The test split keeps its natural
    distribution, because that is the distribution the deployed system faces.
    """
    cap = max_per_class if max_per_class is not None else TRAINING.max_samples_per_class
    if cap is None:
        return frame

    parts = []
    for label, group in frame.groupby(label_column, sort=False):
        if len(group) > cap:
            group = group.sample(n=cap, random_state=random_state)
            logger.info("Downsampled %-12s to %d rows", label, cap)
        parts.append(group)
    balanced = pd.concat(parts, ignore_index=True).sample(frac=1.0, random_state=random_state)
    return balanced.reset_index(drop=True)


def select_feature_columns(frame: pd.DataFrame, restrict_to_core: bool = True) -> list[str]:
    """Choose which raw columns feed the pipeline.

    ``restrict_to_core=True`` limits the model to the 28 features in
    ``CORE_FEATURES``. That is a deliberate trade-off: CIC-IDS2017 offers ~78
    columns, but a compact set keeps the API contract fillable by hand, keeps
    SHAP fast enough for interactive use, and reduces the chance of leaning on
    a capture artefact. Set it to ``False`` to use every numeric column.
    """
    available = set(frame.columns)
    if restrict_to_core:
        chosen = [c for c in CORE_FEATURES if c in available]
        missing = [c for c in CORE_FEATURES if c not in available]
        if missing:
            logger.warning("%d core feature(s) absent from data: %s", len(missing), missing[:8])
        return chosen
    return [
        c
        for c in frame.select_dtypes(include=[np.number]).columns
        if c not in {"label"} and c not in LEAKAGE_COLUMNS
    ]


def build_preprocessor(numeric_features: Iterable[str]) -> Pipeline:
    """Construct the fitted-at-training, replayed-at-inference transformer.

    Structure::

        FunctionTransformer(add_derived_features)   # domain features
                    |
        ColumnTransformer
            numeric  -> SimpleImputer(median) -> RobustScaler
            nominal  -> SimpleImputer(constant) -> OneHotEncoder(ignore)

    Why **median** imputation: flow features are heavily right-skewed, so the
    mean of ``flow_bytes_s`` sits far above the typical flow and would inject a
    misleading value.

    Why **RobustScaler** rather than StandardScaler: it centres on the median
    and scales by the IQR, so a DDoS flow with a packet rate four orders of
    magnitude above normal does not compress every benign flow into a
    near-zero band. Scaling matters for Logistic Regression and for the
    distance-free but still scale-sensitive Isolation Forest; trees are
    invariant to it, but sharing one preprocessor across all models keeps the
    comparison honest.

    Why ``handle_unknown='ignore'`` on the encoder: at inference the system will
    meet port categories or protocols absent from training. Ignoring them
    produces an all-zero block rather than a crash.
    """
    categorical = list(CATEGORICAL_FEATURES)
    # ``protocol`` and ``port_category`` are semantically nominal even though
    # ``protocol`` arrives as an IANA integer, so they are routed to the
    # one-hot branch and must not also appear in the numeric branch.
    numeric = [
        column
        for column in dict.fromkeys([*numeric_features, *DERIVED_FEATURES])
        if column not in categorical
    ]

    numeric_branch = Pipeline(
        steps=[
            ("impute", SimpleImputer(strategy="median")),
            ("scale", RobustScaler(quantile_range=(5.0, 95.0))),
        ]
    )
    categorical_branch = Pipeline(
        steps=[
            ("impute", SimpleImputer(strategy="constant", fill_value="unknown")),
            ("encode", OneHotEncoder(handle_unknown="ignore", sparse_output=False, min_frequency=0.001)),
        ]
    )

    column_transformer = ColumnTransformer(
        transformers=[
            ("numeric", numeric_branch, numeric),
            ("categorical", categorical_branch, categorical),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )

    return Pipeline(
        steps=[
            ("engineer", __import__("src.feature_engineering", fromlist=["x"]).feature_engineering_transformer),
            ("transform", column_transformer),
        ]
    )


def align_features(frame: pd.DataFrame, expected: Iterable[str]) -> pd.DataFrame:
    """Reindex an incoming frame onto the columns the pipeline was fitted with.

    Missing columns are created as NaN and filled by the pipeline's imputer;
    extra columns are dropped. This is what lets the API accept a partial flow
    record without the pipeline raising on unseen columns.
    """
    expected = list(expected)
    out = frame.copy()
    for column in expected:
        if column not in out.columns:
            out[column] = np.nan
    return out[expected]


def prepare_for_inference(frame: pd.DataFrame, expected_columns: Iterable[str]) -> pd.DataFrame:
    """Minimal, inference-safe cleaning: no dedup, no rebalancing, no row drops.

    Deliberately *not* symmetric with :func:`clean_dataset`. At inference we
    must return exactly one prediction per input row, so any operation that
    changes row count is forbidden.
    """
    work = normalize_frame_columns(frame)
    work = work.replace([np.inf, -np.inf], np.nan)
    return align_features(work, expected_columns)


def normalize_frame_columns(frame: pd.DataFrame) -> pd.DataFrame:
    """Snake_case the headers of a user-supplied frame (e.g. an API upload)."""
    from src.data_loader import normalize_columns

    return normalize_columns(frame)


def split_features_labels(frame: pd.DataFrame, label_column: str = "label") -> tuple[pd.DataFrame, pd.Series]:
    """Separate the design matrix from the target vector."""
    if label_column not in frame.columns:
        raise KeyError(f"Label column '{label_column}' not present in frame.")
    return frame.drop(columns=[label_column]), frame[label_column]


def binary_target(labels: pd.Series) -> pd.Series:
    """Collapse the multiclass target to malicious/benign (1/0)."""
    return (labels != BENIGN_LABEL).astype("int8")


def add_derived_preview(frame: pd.DataFrame) -> pd.DataFrame:
    """Expose engineered features for EDA without running the full pipeline."""
    return add_derived_features(frame)
