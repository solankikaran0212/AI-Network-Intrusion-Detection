"""Unsupervised anomaly detection with Isolation Forest.

Why an unsupervised detector at all, when we already have a supervised model?

A supervised classifier can only emit labels it was trained on. Faced with a
genuinely novel attack - a 2026 technique absent from a 2017 capture - it will
confidently assign the nearest known class, most often BENIGN, because benign
traffic dominates the feature space. That is a *silent* false negative, the
worst failure mode in intrusion detection.

Isolation Forest gives an independent, label-free second opinion: it learns what
normal traffic looks like and flags whatever sits far from it. The two models
disagreeing ("classifier says BENIGN, detector says anomalous") is itself a
high-value signal and is exactly what the risk scorer escalates.

Why Isolation Forest specifically:

* It isolates points by random partitioning, so its cost is O(n log n) and it
  scales to millions of flows - One-Class SVM is roughly O(n^2) and is not
  viable on CIC-IDS2017.
* It needs no distance metric, so it tolerates the mixed scales and heavy skew
  of flow features better than LOF or k-NN based detectors.
* It has one meaningful hyper-parameter (``contamination``), which keeps it
  defensible rather than over-fitted.

Training policy: the detector is fitted on **benign flows only**. Fitting on the
full mixture would teach it that attack traffic is part of "normal", defeating
the purpose.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.pipeline import Pipeline

from src.config import BENIGN_LABEL, PATHS, TRAINING, get_logger

logger = get_logger(__name__)


@dataclass
class AnomalyBundle:
    """Serialised anomaly detector plus everything needed to replay it."""

    preprocessor: Pipeline
    detector: IsolationForest
    feature_columns: list[str]
    score_min: float
    score_max: float
    threshold: float
    contamination: float
    trained_on_rows: int

    def to_metadata(self) -> dict[str, Any]:
        return {
            "contamination": self.contamination,
            "threshold": self.threshold,
            "trained_on_rows": self.trained_on_rows,
            "feature_count": len(self.feature_columns),
        }


def train_anomaly_detector(
    frame: pd.DataFrame,
    preprocessor: Pipeline,
    feature_columns: list[str],
    label_column: str = "label",
    contamination: float | None = None,
    random_state: int = TRAINING.random_state,
) -> AnomalyBundle:
    """Fit an Isolation Forest on the benign subset of the training data.

    Args:
        frame: Training rows (already cleaned), including the label column.
        preprocessor: The *already fitted* preprocessing pipeline shared with
            the supervised model. Reusing it guarantees both models see an
            identical feature space, which is what makes their disagreement
            interpretable.
        feature_columns: Raw columns the preprocessor expects.
        label_column: Name of the ground-truth column.
        contamination: Expected proportion of outliers. Defaults to config.

    Returns:
        A fitted :class:`AnomalyBundle` with calibration bounds attached.
    """
    contamination = contamination if contamination is not None else TRAINING.anomaly_contamination

    benign = frame[frame[label_column] == BENIGN_LABEL] if label_column in frame.columns else frame
    if benign.empty:
        logger.warning("No benign rows found; fitting anomaly detector on all rows instead.")
        benign = frame

    features = benign[[c for c in feature_columns if c in benign.columns]]
    matrix = preprocessor.transform(features)

    detector = IsolationForest(
        n_estimators=200,
        max_samples="auto",
        contamination=contamination,
        random_state=random_state,
        n_jobs=TRAINING.n_jobs,
        bootstrap=False,
    )
    detector.fit(matrix)
    logger.info("Isolation Forest fitted on %d benign flows (contamination=%.3f)", len(benign), contamination)

    # Calibrate the score range on a mixed sample so the normalisation covers
    # both normal and attack traffic; otherwise every attack saturates at 1.0.
    calibration = frame.sample(n=min(len(frame), 50_000), random_state=random_state)
    calib_matrix = preprocessor.transform(calibration[[c for c in feature_columns if c in calibration.columns]])
    raw_scores = detector.score_samples(calib_matrix)

    score_min = float(np.percentile(raw_scores, 0.5))
    score_max = float(np.percentile(raw_scores, 99.5))
    threshold = float(detector.offset_)

    return AnomalyBundle(
        preprocessor=preprocessor,
        detector=detector,
        feature_columns=list(feature_columns),
        score_min=score_min,
        score_max=score_max,
        threshold=threshold,
        contamination=float(contamination),
        trained_on_rows=int(len(benign)),
    )


def normalize_scores(raw_scores: np.ndarray, bundle: AnomalyBundle) -> np.ndarray:
    """Convert raw ``score_samples`` output to a 0-1 "how odd is this" scale.

    ``IsolationForest.score_samples`` returns values where **lower means more
    anomalous** (roughly -0.5 to 0.0 in practice). That is unintuitive to expose
    in an API, so we invert and min-max scale against the calibration bounds
    captured at training time.

    The result is a *model-derived anomaly score*, not a probability. A value of
    0.94 does not mean "94% chance this is an attack"; it means "this flow sits
    in the top few percent of oddness relative to the training distribution".
    """
    span = bundle.score_max - bundle.score_min
    if span <= 0:
        return np.zeros_like(raw_scores, dtype=float)
    normalized = (bundle.score_max - raw_scores) / span
    return np.clip(normalized, 0.0, 1.0)


def score_anomalies(frame: pd.DataFrame, bundle: AnomalyBundle) -> tuple[np.ndarray, np.ndarray]:
    """Score flows for anomalousness.

    Returns:
        ``(normalized_scores, flags)`` - scores in [0, 1] where higher is more
        anomalous, and boolean flags from the detector's own decision function.
    """
    from src.preprocessing import prepare_for_inference

    features = prepare_for_inference(frame, bundle.feature_columns)
    matrix = bundle.preprocessor.transform(features)
    raw = bundle.detector.score_samples(matrix)
    flags = bundle.detector.predict(matrix) == -1
    return normalize_scores(raw, bundle), flags


def save_anomaly_bundle(bundle: AnomalyBundle, path: Path | None = None) -> Path:
    """Persist the detector bundle to disk."""
    target = Path(path) if path is not None else PATHS.anomaly_bundle
    target.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, target, compress=3)
    logger.info("Saved anomaly detector to %s", target)
    return target


def load_anomaly_bundle(path: Path | None = None) -> AnomalyBundle:
    """Load a previously saved detector bundle."""
    target = Path(path) if path is not None else PATHS.anomaly_bundle
    if not target.exists():
        raise FileNotFoundError(f"Anomaly detector not found at {target}. Run scripts/train_model.py first.")
    return joblib.load(target)


def anomaly_risk_label(score: float) -> str:
    """Coarse three-band description of an anomaly score, for display only."""
    if score < 0.35:
        return "NORMAL"
    if score < 0.70:
        return "SUSPICIOUS"
    return "CRITICAL"
