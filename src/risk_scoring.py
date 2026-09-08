"""Composite threat/risk scoring.

.. warning::

   The scores produced here are a **portfolio-project heuristic**, not a
   production security standard and not a calibrated probability of compromise.
   Real SOC risk models incorporate asset criticality, business context, threat
   intelligence, kill-chain stage and analyst feedback loops - none of which
   exist in a flow-level dataset. Treat these numbers as a *triage ordering*,
   which is what they are useful for: they answer "which of these 10,000 flows
   should a human look at first?", not "what is the probability this host is
   compromised?".

Score construction (0-100), a weighted blend of four independent signals:

===========================  ======  ==================================================
Component                    Weight  Rationale
===========================  ======  ==================================================
Class severity               0.45    A Botnet beacon deserves attention over a PortScan.
Model confidence             0.25    A 0.51-confidence call is weaker evidence than 0.99.
Anomaly score                0.20    Catches novel attacks the classifier never saw.
Traffic characteristics      0.10    Volumetric/behavioural red flags, model-independent.
===========================  ======  ==================================================

The weights are a judgement call, exposed as constants so they can be argued
about and tuned rather than buried in a formula.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Any, Mapping

import numpy as np
import pandas as pd

from src.config import BENIGN_LABEL, CLASS_SEVERITY, get_logger

logger = get_logger(__name__)

WEIGHT_SEVERITY = 0.45
WEIGHT_CONFIDENCE = 0.25
WEIGHT_ANOMALY = 0.20
WEIGHT_TRAFFIC = 0.10

#: Bands used by the API, dashboard and alerting logic.
RISK_BANDS: tuple[tuple[int, int, str], ...] = (
    (0, 30, "LOW"),
    (31, 60, "MEDIUM"),
    (61, 80, "HIGH"),
    (81, 100, "CRITICAL"),
)

#: Empirical thresholds above which a flow looks volumetrically abnormal.
#: Derived from the benign quantiles of the training corpus, not from a
#: published standard.
_PPS_ALERT = 5_000.0
_BPS_ALERT = 5_000_000.0
_FLAG_DENSITY_ALERT = 0.9
_ASYMMETRY_ALERT = 50.0


@dataclass
class RiskAssessment:
    """Full, auditable breakdown of a single flow's risk score."""

    risk_score: int
    risk_level: str
    predicted_class: str
    confidence: float
    anomaly_score: float
    is_anomaly: bool
    components: dict[str, float]
    triggered_indicators: list[str]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def risk_level_for(score: float) -> str:
    """Map a 0-100 score onto its named band."""
    value = int(round(float(np.clip(score, 0, 100))))
    for low, high, name in RISK_BANDS:
        if low <= value <= high:
            return name
    return "CRITICAL" if value > 100 else "LOW"


def _severity_component(predicted_class: str) -> float:
    """Base severity of the predicted attack category, 0-100."""
    return float(CLASS_SEVERITY.get(predicted_class, 60 if predicted_class != BENIGN_LABEL else 0))


def _confidence_component(predicted_class: str, confidence: float) -> float:
    """Confidence rescaled so it can only *modulate*, never create, risk.

    For benign predictions the relationship inverts: high confidence that a flow
    is benign should *lower* risk, so we return the complement. This stops a
    confidently-benign flow inheriting a mid-range confidence score.
    """
    conf = float(np.clip(confidence, 0.0, 1.0))
    if predicted_class == BENIGN_LABEL:
        return (1.0 - conf) * 100.0
    # Rescale [0.5, 1.0] -> [0, 100]: below 0.5 the multiclass call is barely
    # better than the prior, so it contributes no confidence-driven risk.
    return float(np.clip((conf - 0.5) / 0.5, 0.0, 1.0) * 100.0)


def _traffic_component(flow: Mapping[str, Any] | pd.Series) -> tuple[float, list[str]]:
    """Model-independent red flags read straight off the flow record.

    Kept separate from the classifier on purpose: if the model is wrong or
    encounters a class it was never trained on, these indicators still fire.
    """
    indicators: list[str] = []
    score = 0.0

    def _get(key: str, default: float = 0.0) -> float:
        try:
            value = float(flow.get(key, default))  # type: ignore[union-attr]
            return default if not np.isfinite(value) else value
        except (TypeError, ValueError):
            return default

    pps = _get("packets_per_second")
    bps = _get("bytes_per_second")
    flag_density = _get("tcp_flag_density")
    asymmetry = _get("fwd_bwd_packet_ratio")
    bwd_packets = _get("total_backward_packets")
    fwd_packets = _get("total_fwd_packets")

    if pps > _PPS_ALERT:
        score += 30.0
        indicators.append(f"High packet rate ({pps:,.0f} pkt/s)")
    if bps > _BPS_ALERT:
        score += 25.0
        indicators.append(f"High byte rate ({bps / 1e6:,.1f} MB/s)")
    if flag_density > _FLAG_DENSITY_ALERT:
        score += 20.0
        indicators.append("Control-flag-dominated flow (near-zero payload)")
    if asymmetry > _ASYMMETRY_ALERT:
        score += 15.0
        indicators.append(f"Extreme directional asymmetry ({asymmetry:,.0f}:1 fwd/bwd)")
    if bwd_packets == 0 and fwd_packets > 2:
        score += 15.0
        indicators.append("No response traffic - probe or half-open connection")

    return float(np.clip(score, 0.0, 100.0)), indicators


def _anomaly_component(anomaly_score: float) -> float:
    """Normalised anomaly contribution, already 0-1 from the detector."""
    return float(np.clip(anomaly_score, 0.0, 1.0)) * 100.0


def compute_risk(
    predicted_class: str,
    confidence: float,
    anomaly_score: float = 0.0,
    is_anomaly: bool = False,
    flow: Mapping[str, Any] | pd.Series | None = None,
) -> RiskAssessment:
    """Blend the four signals into one auditable assessment.

    Args:
        predicted_class: Winning class from the supervised model.
        confidence: Predicted probability of that class, 0-1.
        anomaly_score: Normalised Isolation Forest score, 0-1 (higher = odder).
        is_anomaly: Hard flag from the detector's own threshold.
        flow: The original flow record, used for the traffic component.

    Returns:
        A :class:`RiskAssessment` with the final score, band, per-component
        breakdown and human-readable indicators.
    """
    flow = flow if flow is not None else {}

    severity = _severity_component(predicted_class)
    conf_component = _confidence_component(predicted_class, confidence)
    anomaly_component = _anomaly_component(anomaly_score)
    traffic_component, indicators = _traffic_component(flow)

    raw = (
        WEIGHT_SEVERITY * severity
        + WEIGHT_CONFIDENCE * conf_component
        + WEIGHT_ANOMALY * anomaly_component
        + WEIGHT_TRAFFIC * traffic_component
    )

    # A confidently-benign flow that the anomaly detector also considers normal
    # should sit firmly in LOW, regardless of incidental traffic indicators.
    if predicted_class == BENIGN_LABEL and not is_anomaly:
        raw *= 0.5

    # Conversely, a flow the detector flags as anomalous cannot be dismissed
    # outright even if the classifier calls it benign - this is the "unknown
    # attack" path the Isolation Forest exists to cover.
    if is_anomaly:
        raw = max(raw, 45.0)
        indicators.append("Flagged as anomalous by unsupervised detector")

    score = int(round(float(np.clip(raw, 0, 100))))

    return RiskAssessment(
        risk_score=score,
        risk_level=risk_level_for(score),
        predicted_class=predicted_class,
        confidence=round(float(confidence), 4),
        anomaly_score=round(float(anomaly_score), 4),
        is_anomaly=bool(is_anomaly),
        components={
            "class_severity": round(severity, 2),
            "model_confidence": round(conf_component, 2),
            "anomaly": round(anomaly_component, 2),
            "traffic_characteristics": round(traffic_component, 2),
        },
        triggered_indicators=indicators,
    )


def compute_risk_batch(
    predictions: np.ndarray | list[str],
    confidences: np.ndarray | list[float],
    anomaly_scores: np.ndarray | list[float] | None = None,
    anomaly_flags: np.ndarray | list[bool] | None = None,
    flows: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Vectorised-ish batch scoring. Returns one row per input flow."""
    n = len(predictions)
    anomaly_scores = np.zeros(n) if anomaly_scores is None else np.asarray(anomaly_scores, dtype=float)
    anomaly_flags = np.zeros(n, dtype=bool) if anomaly_flags is None else np.asarray(anomaly_flags, dtype=bool)

    records = []
    for i in range(n):
        flow_row = flows.iloc[i] if flows is not None and i < len(flows) else None
        assessment = compute_risk(
            predicted_class=str(predictions[i]),
            confidence=float(confidences[i]),
            anomaly_score=float(anomaly_scores[i]),
            is_anomaly=bool(anomaly_flags[i]),
            flow=flow_row,
        )
        records.append(
            {
                "prediction": assessment.predicted_class,
                "confidence": assessment.confidence,
                "risk_score": assessment.risk_score,
                "risk_level": assessment.risk_level,
                "anomaly_score": assessment.anomaly_score,
                "is_anomaly": assessment.is_anomaly,
                "indicators": "; ".join(assessment.triggered_indicators),
            }
        )
    return pd.DataFrame.from_records(records)
