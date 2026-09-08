"""Supervised model training, comparison and selection.

Protocol used here, and why each piece exists:

**Three-way split (train / validation / test).** The test set is opened exactly
once, at the very end, to report final numbers. All model comparison and
selection happens on the validation split. Selecting a model on the test set
would make the reported test score optimistically biased - you have used the
test labels to make a decision, so it is no longer held out.

**Stratified splitting.** With classes ranging from ~1,500 to ~2M rows, a random
split can leave a rare class entirely absent from the test set. Stratification
preserves class proportions in every split.

**Preprocessor fitted on training data only.** The scaler's median and IQR, and
the encoder's category vocabulary, are learned from the training rows and then
*applied* to validation and test. Fitting on the full dataset first would leak
distributional information from the held-out data into training.

**Class weighting instead of synthetic oversampling.** See
``preprocessing.balance_classes`` for the argument against SMOTE on flow data.
"""

from __future__ import annotations

import json
import platform
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.class_weight import compute_sample_weight

from src.anomaly_detection import save_anomaly_bundle, train_anomaly_detector
from src.config import PATHS, SERVICE, TRAINING, get_logger
from src.evaluate import EvaluationResult, comparison_table, evaluate_model, save_metrics
from src.preprocessing import balance_classes, build_preprocessor, select_feature_columns

logger = get_logger(__name__)

try:
    from xgboost import XGBClassifier

    XGBOOST_AVAILABLE = True
except ImportError:  # pragma: no cover - optional dependency
    XGBOOST_AVAILABLE = False
    logger.warning("xgboost not installed; that model will be skipped.")


@dataclass
class ModelBundle:
    """Everything needed to reproduce a prediction, in one serialisable object.

    Shipping the preprocessor *inside* the same artifact as the classifier is
    deliberate. If they were saved separately, nothing would stop someone
    loading v2 of the model with v1 of the scaler - silent, hard-to-debug
    train/serve skew. One file, one version, one consistent transformation.
    """

    preprocessor: Pipeline
    model: Any
    label_encoder: LabelEncoder
    feature_columns: list[str]
    class_names: list[str]
    model_name: str
    model_version: str
    trained_at: str
    training_rows: int
    metrics: dict[str, Any] = field(default_factory=dict)
    environment: dict[str, str] = field(default_factory=dict)

    @property
    def feature_count(self) -> int:
        return len(self.feature_columns)

    def transformed_feature_names(self) -> list[str]:
        """Names of the columns the classifier actually sees, post-transform."""
        try:
            return list(self.preprocessor.named_steps["transform"].get_feature_names_out())
        except Exception:  # noqa: BLE001
            return [f"feature_{i}" for i in range(getattr(self.model, "n_features_in_", 0))]

    def to_metadata(self) -> dict[str, Any]:
        return {
            "model_name": self.model_name,
            "model_version": self.model_version,
            "trained_at": self.trained_at,
            "training_rows": self.training_rows,
            "supported_classes": self.class_names,
            "feature_count": self.feature_count,
            "raw_features": self.feature_columns,
            "transformed_feature_count": len(self.transformed_feature_names()),
            "environment": self.environment,
            "metrics": self.metrics,
        }


def build_candidate_models(n_classes: int, random_state: int = TRAINING.random_state) -> dict[str, Any]:
    """Instantiate the three candidate classifiers.

    **Logistic Regression** - the baseline. Linear, fast, fully interpretable
    via coefficients. If a complex model cannot beat it, the complexity is not
    earning its keep. ``saga`` handles the multinomial case and scales to large
    sparse-ish matrices.

    **Random Forest** - bagged trees. Captures the non-linear threshold
    behaviour that defines network attacks ("packet rate above X *and*
    asymmetry above Y"), resists overfitting through averaging, and gives
    built-in feature importances. ``class_weight='balanced_subsample'``
    reweights within each bootstrap sample, which matters at our imbalance.

    **XGBoost** - gradient boosting. Fits residuals sequentially, so it usually
    edges out bagging on tabular data, and it is the standard strong baseline
    for this problem class. Trades some training speed and interpretability for
    that accuracy.
    """
    models: dict[str, Any] = {
        "LogisticRegression": LogisticRegression(
            max_iter=1000,
            solver="saga",
            class_weight="balanced",
            random_state=random_state,
            C=1.0,
        ),
        "RandomForest": RandomForestClassifier(
            n_estimators=200,
            max_depth=None,
            min_samples_split=5,
            min_samples_leaf=2,
            max_features="sqrt",
            class_weight="balanced_subsample",
            n_jobs=TRAINING.n_jobs,
            random_state=random_state,
        ),
    }
    if XGBOOST_AVAILABLE:
        models["XGBoost"] = XGBClassifier(
            n_estimators=300,
            max_depth=8,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_lambda=1.0,
            objective="multi:softprob" if n_classes > 2 else "binary:logistic",
            num_class=n_classes if n_classes > 2 else None,
            tree_method="hist",
            eval_metric="mlogloss",
            n_jobs=TRAINING.n_jobs,
            random_state=random_state,
            verbosity=0,
        )
    return models


def split_data(
    frame: pd.DataFrame,
    label_column: str = "label",
    test_size: float = TRAINING.test_size,
    val_size: float = TRAINING.validation_size,
    random_state: int = TRAINING.random_state,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Stratified three-way split: train / validation / test.

    The validation fraction is taken *from the training remainder*, so
    ``val_size=0.2`` with ``test_size=0.2`` yields 64/16/20.
    """
    train_val, test = train_test_split(
        frame,
        test_size=test_size,
        stratify=frame[label_column],
        random_state=random_state,
        shuffle=True,
    )
    train, validation = train_test_split(
        train_val,
        test_size=val_size,
        stratify=train_val[label_column],
        random_state=random_state,
        shuffle=True,
    )
    logger.info(
        "Split -> train=%d  validation=%d  test=%d", len(train), len(validation), len(test)
    )
    return (
        train.reset_index(drop=True),
        validation.reset_index(drop=True),
        test.reset_index(drop=True),
    )


def train_models(
    frame: pd.DataFrame,
    label_column: str = "label",
    restrict_to_core: bool = True,
    run_cross_validation: bool = False,
    balance: bool = True,
) -> tuple[ModelBundle, list[EvaluationResult], dict[str, Any]]:
    """Train, compare and select the best supervised model.

    Args:
        frame: Cleaned dataset including the label column.
        label_column: Ground-truth column name.
        restrict_to_core: Use the compact core feature set (see preprocessing).
        run_cross_validation: Additionally run stratified k-fold CV on the
            training split. Off by default because it multiplies training time.
        balance: Apply majority-class downsampling to the training split only.

    Returns:
        ``(best_bundle, validation_results, training_report)``.
    """
    started = time.perf_counter()
    train_df, val_df, test_df = split_data(frame, label_column=label_column)

    if balance:
        before = len(train_df)
        train_df = balance_classes(train_df, label_column=label_column)
        logger.info("Balanced training split: %d -> %d rows", before, len(train_df))

    feature_columns = select_feature_columns(train_df.drop(columns=[label_column]), restrict_to_core)
    if not feature_columns:
        raise ValueError("No usable feature columns were selected; check the input dataset.")
    logger.info("Using %d raw feature column(s)", len(feature_columns))

    X_train, y_train = train_df[feature_columns], train_df[label_column]
    X_val, y_val = val_df[feature_columns], val_df[label_column]
    X_test, y_test = test_df[feature_columns], test_df[label_column]

    # Fit the preprocessor on training data ONLY, then reuse it everywhere.
    preprocessor = build_preprocessor(feature_columns)
    train_matrix = preprocessor.fit_transform(X_train)
    val_matrix = preprocessor.transform(X_val)
    test_matrix = preprocessor.transform(X_test)
    logger.info("Transformed feature space: %d columns", train_matrix.shape[1])

    encoder = LabelEncoder().fit(sorted(frame[label_column].unique()))
    class_names = list(encoder.classes_)
    y_train_enc = encoder.transform(y_train)

    sample_weight = compute_sample_weight(class_weight="balanced", y=y_train_enc)

    candidates = build_candidate_models(n_classes=len(class_names))
    validation_results: list[EvaluationResult] = []
    fitted: dict[str, Any] = {}
    timings: dict[str, float] = {}
    cv_scores: dict[str, dict[str, float]] = {}

    for name, model in candidates.items():
        logger.info("Training %s ...", name)
        fit_start = time.perf_counter()
        try:
            if name == "XGBoost":
                model.fit(train_matrix, y_train_enc, sample_weight=sample_weight)
            else:
                model.fit(train_matrix, y_train_enc)
        except Exception:  # noqa: BLE001 - one failing model must not kill the run
            logger.exception("Training failed for %s; skipping.", name)
            continue
        timings[name] = time.perf_counter() - fit_start

        y_val_pred = encoder.inverse_transform(model.predict(val_matrix))
        y_val_proba = model.predict_proba(val_matrix) if hasattr(model, "predict_proba") else None

        result = evaluate_model(name, y_val, y_val_pred, y_val_proba, class_names, split="validation")
        validation_results.append(result)
        fitted[name] = model
        logger.info("  %s (%.1fs)", result.summary_line(), timings[name])

        if run_cross_validation:
            folds = StratifiedKFold(n_splits=TRAINING.cv_folds, shuffle=True, random_state=TRAINING.random_state)
            scores = cross_val_score(
                model, train_matrix, y_train_enc, cv=folds, scoring="f1_macro", n_jobs=1
            )
            cv_scores[name] = {"mean_macro_f1": float(scores.mean()), "std_macro_f1": float(scores.std())}
            logger.info("  CV macro-F1: %.4f +/- %.4f", scores.mean(), scores.std())

    if not validation_results:
        raise RuntimeError("Every candidate model failed to train; see logs.")

    # ---- Selection happens on VALIDATION, never on test. ------------------- #
    best_result = max(validation_results, key=lambda r: r.macro_f1)
    best_name = best_result.model_name
    best_model = fitted[best_name]
    logger.info("Selected %s on validation macro-F1 = %.4f", best_name, best_result.macro_f1)

    # ---- Test set is opened exactly once, here. ---------------------------- #
    y_test_pred = encoder.inverse_transform(best_model.predict(test_matrix))
    y_test_proba = best_model.predict_proba(test_matrix) if hasattr(best_model, "predict_proba") else None
    test_result = evaluate_model(best_name, y_test, y_test_pred, y_test_proba, class_names, split="test")
    logger.info("FINAL TEST -> %s", test_result.summary_line())

    bundle = ModelBundle(
        preprocessor=preprocessor,
        model=best_model,
        label_encoder=encoder,
        feature_columns=feature_columns,
        class_names=class_names,
        model_name=best_name,
        model_version=SERVICE.model_version,
        trained_at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
        training_rows=int(len(train_df)),
        metrics={"validation": best_result.to_dict(), "test": test_result.to_dict()},
        environment={
            "python": platform.python_version(),
            "scikit_learn": sklearn.__version__,
            "numpy": np.__version__,
            "pandas": pd.__version__,
        },
    )

    report: dict[str, Any] = {
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "selected_model": best_name,
        "selection_metric": TRAINING.selection_metric,
        "selection_split": "validation",
        "dataset": {
            "total_rows": int(len(frame)),
            "train_rows": int(len(train_df)),
            "validation_rows": int(len(val_df)),
            "test_rows": int(len(test_df)),
            "classes": class_names,
            "class_distribution": frame[label_column].value_counts().to_dict(),
        },
        "features": {
            "raw_count": len(feature_columns),
            "raw_columns": feature_columns,
            "transformed_count": int(train_matrix.shape[1]),
        },
        "training_seconds": {k: round(v, 2) for k, v in timings.items()},
        "cross_validation": cv_scores,
        "validation_results": [r.to_dict() for r in validation_results],
        "test_result": test_result.to_dict(),
        "comparison_table": comparison_table(validation_results).to_dict(orient="records"),
        "total_runtime_seconds": round(time.perf_counter() - started, 2),
    }

    # Anomaly detector shares the fitted preprocessor and the same training rows.
    try:
        anomaly_bundle = train_anomaly_detector(
            frame=train_df, preprocessor=preprocessor, feature_columns=feature_columns, label_column=label_column
        )
        save_anomaly_bundle(anomaly_bundle)
        report["anomaly_detector"] = anomaly_bundle.to_metadata()
    except Exception:  # noqa: BLE001
        logger.exception("Anomaly detector training failed; continuing without it.")

    return bundle, validation_results, report


def save_model_bundle(bundle: ModelBundle, path: Path | None = None) -> Path:
    """Serialise the model bundle with joblib."""
    target = Path(path) if path is not None else PATHS.model_bundle
    target.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, target, compress=3)
    size_mb = target.stat().st_size / (1024 * 1024)
    logger.info("Saved model bundle to %s (%.1f MB)", target, size_mb)

    metadata_path = target.with_suffix(".metadata.json")
    with metadata_path.open("w", encoding="utf-8") as handle:
        json.dump(bundle.to_metadata(), handle, indent=2, default=str)
    return target


def load_model_bundle(path: Path | None = None) -> ModelBundle:
    """Load a serialised model bundle."""
    target = Path(path) if path is not None else PATHS.model_bundle
    if not target.exists():
        raise FileNotFoundError(
            f"Model bundle not found at {target}. Train one with `python scripts/train_model.py`."
        )
    return joblib.load(target)


def run_training(
    frame: pd.DataFrame,
    label_column: str = "label",
    restrict_to_core: bool = True,
    run_cross_validation: bool = False,
) -> tuple[ModelBundle, dict[str, Any]]:
    """End-to-end training entrypoint: train, select, persist, report."""
    PATHS.ensure()
    bundle, _results, report = train_models(
        frame,
        label_column=label_column,
        restrict_to_core=restrict_to_core,
        run_cross_validation=run_cross_validation,
    )
    save_model_bundle(bundle)
    save_metrics(report)
    return bundle, report
