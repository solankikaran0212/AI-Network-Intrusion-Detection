"""Inference service: the single entry point shared by the API and dashboard.

Having exactly one prediction path is an architectural decision, not an
accident. If the API and the dashboard each did their own preprocessing, they
would eventually disagree, and the bug would only surface in whichever surface
gets less testing. Both import :class:`ThreatPredictor` instead.

The predictor composes three independently-testable pieces:

1. Supervised classifier - "which known attack class is this?"
2. Isolation Forest - "does this look like anything we have seen before?"
3. Risk scorer - "how urgently should a human look at it?"
"""

from __future__ import annotations

import threading
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from src.anomaly_detection import AnomalyBundle, load_anomaly_bundle, score_anomalies
from src.config import BENIGN_LABEL, get_logger
from src.feature_engineering import add_derived_features
from src.preprocessing import prepare_for_inference
from src.risk_scoring import RiskAssessment, compute_risk
from src.train import ModelBundle, load_model_bundle

logger = get_logger(__name__)


@dataclass
class PredictionResult:
    """One scored flow, ready to serialise."""

    prediction: str
    confidence: float
    risk_score: int
    risk_level: str
    is_anomaly: bool
    anomaly_score: float
    is_malicious: bool
    class_probabilities: dict[str, float]
    risk_components: dict[str, float]
    indicators: list[str]
    top_features: list[dict[str, Any]]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class ThreatPredictor:
    """Loads the artifacts once and scores flows.

    Thread-safe for read access: artifacts are immutable after load and the
    scikit-learn estimators used here have no mutable predict-time state. The
    lock only guards lazy initialisation.
    """

    _lock = threading.Lock()

    def __init__(
        self,
        model_bundle: ModelBundle | None = None,
        anomaly_bundle: AnomalyBundle | None = None,
        model_path: Path | None = None,
        anomaly_path: Path | None = None,
        enable_explanations: bool = True,
        use_anomaly_detector: bool = True,
    ) -> None:
        """
        Args:
            model_bundle: A pre-loaded bundle. When omitted, one is read from
                ``model_path`` (or the configured default).
            anomaly_bundle: A pre-loaded detector.
            model_path: Override for the model artifact location.
            anomaly_path: Override for the detector artifact location.
            enable_explanations: Build a SHAP explainer on first use.
            use_anomaly_detector: Set ``False`` to run supervised-only. This is
                an explicit switch because ``anomaly_bundle=None`` alone is
                ambiguous - it cannot distinguish "not supplied, load the
                default" from "deliberately disabled".
        """
        self.bundle = model_bundle if model_bundle is not None else load_model_bundle(model_path)
        self.enable_explanations = enable_explanations
        self._explainer = None

        if not use_anomaly_detector:
            self.anomaly = None
            logger.info("Anomaly detector disabled by caller; anomaly scores will be 0.")
        elif anomaly_bundle is not None:
            self.anomaly = anomaly_bundle
        else:
            try:
                self.anomaly = load_anomaly_bundle(anomaly_path)
            except FileNotFoundError:
                logger.warning("No anomaly detector found; anomaly scores will be 0.")
                self.anomaly = None

        logger.info(
            "ThreatPredictor ready: model=%s version=%s classes=%d features=%d",
            self.bundle.model_name,
            self.bundle.model_version,
            len(self.bundle.class_names),
            self.bundle.feature_count,
        )

    # ---------------------------------------------------------------- helpers

    @property
    def feature_columns(self) -> list[str]:
        return list(self.bundle.feature_columns)

    @property
    def class_names(self) -> list[str]:
        return list(self.bundle.class_names)

    def _explainer_instance(self):
        """Lazily build the SHAP explainer - it is expensive to construct."""
        if not self.enable_explanations:
            return None
        if self._explainer is None:
            with self._lock:
                if self._explainer is None:
                    from src.explainability import ModelExplainer

                    try:
                        self._explainer = ModelExplainer(self.bundle)
                    except Exception:  # noqa: BLE001
                        logger.exception("Explainer unavailable; predictions will omit explanations.")
                        self.enable_explanations = False
                        return None
        return self._explainer

    def _to_frame(self, flow: Mapping[str, Any] | pd.Series | pd.DataFrame) -> pd.DataFrame:
        if isinstance(flow, pd.DataFrame):
            return flow.copy()
        if isinstance(flow, pd.Series):
            return flow.to_frame().T
        return pd.DataFrame([dict(flow)])

    # ------------------------------------------------------------- prediction

    def predict_frame(
        self, frame: pd.DataFrame, explain: bool = False, top_k: int = 5
    ) -> list[PredictionResult]:
        """Score a batch of flows.

        Args:
            frame: One row per flow. Missing columns are imputed; extra columns
                are ignored.
            explain: Attach per-row top contributing features. Costly, so the
                API enables it only for single predictions.
            top_k: Number of contributing features to report.

        Returns:
            One :class:`PredictionResult` per input row, in input order.
        """
        if frame.empty:
            return []

        features = prepare_for_inference(frame, self.bundle.feature_columns)
        matrix = self.bundle.preprocessor.transform(features)

        encoded = self.bundle.model.predict(matrix)
        predictions = self.bundle.label_encoder.inverse_transform(encoded)

        if hasattr(self.bundle.model, "predict_proba"):
            probabilities = self.bundle.model.predict_proba(matrix)
        else:  # pragma: no cover - all configured models expose predict_proba
            probabilities = np.zeros((len(frame), len(self.class_names)))
            probabilities[np.arange(len(frame)), encoded] = 1.0
        confidences = probabilities.max(axis=1)

        if self.anomaly is not None:
            anomaly_scores, anomaly_flags = score_anomalies(features, self.anomaly)
        else:
            anomaly_scores = np.zeros(len(frame))
            anomaly_flags = np.zeros(len(frame), dtype=bool)

        # Derived features are needed by the traffic component of the risk score.
        enriched = add_derived_features(features)

        explanations: list[list[dict[str, Any]]] = [[] for _ in range(len(frame))]
        if explain:
            explainer = self._explainer_instance()
            if explainer is not None:
                try:
                    explanations = explainer.explain_batch(features, matrix, encoded, top_k=top_k)
                except Exception:  # noqa: BLE001 - explanation is never load-bearing
                    logger.exception("Explanation failed; returning predictions without it.")

        results: list[PredictionResult] = []
        for i, prediction in enumerate(predictions):
            assessment: RiskAssessment = compute_risk(
                predicted_class=str(prediction),
                confidence=float(confidences[i]),
                anomaly_score=float(anomaly_scores[i]),
                is_anomaly=bool(anomaly_flags[i]),
                flow=enriched.iloc[i],
            )
            results.append(
                PredictionResult(
                    prediction=str(prediction),
                    confidence=round(float(confidences[i]), 4),
                    risk_score=assessment.risk_score,
                    risk_level=assessment.risk_level,
                    is_anomaly=bool(anomaly_flags[i]),
                    anomaly_score=round(float(anomaly_scores[i]), 4),
                    is_malicious=str(prediction) != BENIGN_LABEL,
                    class_probabilities={
                        name: round(float(probabilities[i][j]), 4)
                        for j, name in enumerate(self.class_names)
                    },
                    risk_components=assessment.components,
                    indicators=assessment.triggered_indicators,
                    top_features=explanations[i],
                )
            )
        return results

    def predict_one(
        self, flow: Mapping[str, Any] | pd.Series, explain: bool = True, top_k: int = 5
    ) -> PredictionResult:
        """Score a single flow record."""
        results = self.predict_frame(self._to_frame(flow), explain=explain, top_k=top_k)
        if not results:
            raise ValueError("Prediction produced no result for the supplied flow.")
        return results[0]

    def predict_csv(self, frame: pd.DataFrame, explain: bool = False) -> pd.DataFrame:
        """Score an uploaded CSV, returning a tabular result with a summary.

        The original columns are preserved alongside the prediction columns so
        an analyst can pivot on source data without a second join.
        """
        results = self.predict_frame(frame, explain=explain)
        scored = pd.DataFrame(
            {
                "prediction": [r.prediction for r in results],
                "confidence": [r.confidence for r in results],
                "risk_score": [r.risk_score for r in results],
                "risk_level": [r.risk_level for r in results],
                "anomaly_score": [r.anomaly_score for r in results],
                "is_anomaly": [r.is_anomaly for r in results],
                "is_malicious": [r.is_malicious for r in results],
                "indicators": ["; ".join(r.indicators) for r in results],
            },
            index=frame.index[: len(results)],
        )
        return pd.concat([frame.reset_index(drop=True), scored.reset_index(drop=True)], axis=1)

    # ---------------------------------------------------------------- summary

    @staticmethod
    def summarize(scored: pd.DataFrame) -> dict[str, Any]:
        """Aggregate a scored frame into SOC-style headline statistics."""
        total = int(len(scored))
        if total == 0:
            return {"total_records": 0}

        malicious = int(scored["is_malicious"].sum())
        return {
            "total_records": total,
            "benign_records": total - malicious,
            "malicious_records": malicious,
            "attack_rate": round(malicious / total, 4),
            "anomalies_detected": int(scored["is_anomaly"].sum()),
            "critical_threats": int((scored["risk_level"] == "CRITICAL").sum()),
            "high_threats": int((scored["risk_level"] == "HIGH").sum()),
            "mean_risk_score": round(float(scored["risk_score"].mean()), 2),
            "max_risk_score": int(scored["risk_score"].max()),
            "attack_distribution": scored["prediction"].value_counts().to_dict(),
            "risk_distribution": scored["risk_level"].value_counts().to_dict(),
            "mean_confidence": round(float(scored["confidence"].mean()), 4),
        }

    def model_info(self) -> dict[str, Any]:
        """Metadata for the ``GET /model-info`` endpoint."""
        info = self.bundle.to_metadata()
        info["anomaly_detector_loaded"] = self.anomaly is not None
        info["explanations_enabled"] = self.enable_explanations
        if self.anomaly is not None:
            info["anomaly_detector"] = self.anomaly.to_metadata()
        return info


_PREDICTOR: ThreatPredictor | None = None
_PREDICTOR_LOCK = threading.Lock()


def get_predictor(force_reload: bool = False, **kwargs: Any) -> ThreatPredictor:
    """Process-wide singleton so model artifacts load once, not per request."""
    global _PREDICTOR
    if _PREDICTOR is None or force_reload:
        with _PREDICTOR_LOCK:
            if _PREDICTOR is None or force_reload:
                _PREDICTOR = ThreatPredictor(**kwargs)
    return _PREDICTOR


def reset_predictor() -> None:
    """Clear the singleton. Used by tests to isolate fixtures."""
    global _PREDICTOR
    with _PREDICTOR_LOCK:
        _PREDICTOR = None


def example_flow(kind: str = "benign") -> dict[str, float]:
    """Return a hand-built example flow, for API docs, tests and the dashboard.

    These are *illustrative inputs*, not recorded traffic, and carry no claim
    about what the model will predict for them.
    """
    templates: dict[str, dict[str, float]] = {
        "benign": {
            "destination_port": 443,
            "protocol": 6,
            "flow_duration": 5_000_000,
            "total_fwd_packets": 24,
            "total_backward_packets": 22,
            "total_length_of_fwd_packets": 3_100,
            "total_length_of_bwd_packets": 18_500,
            "fwd_packet_length_mean": 129.2,
            "bwd_packet_length_mean": 840.9,
            "flow_bytes_s": 4_320.0,
            "flow_packets_s": 9.2,
            "packet_length_mean": 469.6,
            "packet_length_std": 402.1,
            "syn_flag_count": 1,
            "ack_flag_count": 20,
            "fin_flag_count": 1,
            "psh_flag_count": 6,
            "rst_flag_count": 0,
            "urg_flag_count": 0,
            "min_packet_length": 0,
            "max_packet_length": 1460,
            "down_up_ratio": 1,
            "average_packet_size": 469.6,
            "init_win_bytes_forward": 29200,
            "init_win_bytes_backward": 26883,
        },
        "scan": {
            "destination_port": 3389,
            "protocol": 6,
            "flow_duration": 120,
            "total_fwd_packets": 2,
            "total_backward_packets": 0,
            "total_length_of_fwd_packets": 0,
            "total_length_of_bwd_packets": 0,
            "fwd_packet_length_mean": 0.0,
            "bwd_packet_length_mean": 0.0,
            "flow_bytes_s": 0.0,
            "flow_packets_s": 16_666.0,
            "packet_length_mean": 0.0,
            "packet_length_std": 0.0,
            "syn_flag_count": 2,
            "ack_flag_count": 0,
            "fin_flag_count": 0,
            "psh_flag_count": 0,
            "rst_flag_count": 0,
            "urg_flag_count": 0,
            "min_packet_length": 0,
            "max_packet_length": 0,
            "down_up_ratio": 0,
            "average_packet_size": 0.0,
            "init_win_bytes_forward": 1024,
            "init_win_bytes_backward": -1,
        },
        "flood": {
            "destination_port": 80,
            "protocol": 6,
            "flow_duration": 3_000,
            "total_fwd_packets": 480,
            "total_backward_packets": 1,
            "total_length_of_fwd_packets": 28_800,
            "total_length_of_bwd_packets": 0,
            "fwd_packet_length_mean": 60.0,
            "bwd_packet_length_mean": 0.0,
            "flow_bytes_s": 9_600_000.0,
            "flow_packets_s": 160_000.0,
            "packet_length_mean": 60.0,
            "packet_length_std": 2.0,
            "syn_flag_count": 480,
            "ack_flag_count": 0,
            "fin_flag_count": 0,
            "psh_flag_count": 0,
            "rst_flag_count": 1,
            "urg_flag_count": 0,
            "min_packet_length": 60,
            "max_packet_length": 64,
            "down_up_ratio": 0,
            "average_packet_size": 60.0,
            "init_win_bytes_forward": 512,
            "init_win_bytes_backward": -1,
        },
    }
    return templates.get(kind, templates["benign"]).copy()
