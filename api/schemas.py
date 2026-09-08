"""Pydantic models defining the API contract.

Keeping the schemas separate from the routing logic in ``main.py`` means the
contract can be read, reviewed and imported without pulling in the application.

Validation philosophy: an intrusion detection system is exposed to hostile
input by definition, so every numeric field is bounded rather than merely
typed. A negative packet count or a port above 65535 is rejected at the edge
with a 422 instead of being quietly imputed and scored.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

# Upper bounds are deliberately generous - they exist to reject impossible
# values (negative counts, ports above the 16-bit range) rather than to encode
# an assumption about what "normal" traffic looks like.
_MAX_COUNT = 1e9
_MAX_BYTES = 1e12
_MAX_RATE = 1e12


class NetworkFlowRequest(BaseModel):
    """A single bidirectional network flow, as emitted by a flow meter.

    Every field except ``destination_port`` is optional: real collectors export
    different subsets, and the preprocessing pipeline imputes anything missing
    using the medians learned at training time. Unknown extra fields are
    accepted and ignored, so a full 78-column CICFlowMeter record can be posted
    without trimming it first.
    """

    model_config = ConfigDict(
        extra="allow",
        json_schema_extra={
            "example": {
                "destination_port": 80,
                "protocol": 6,
                "flow_duration": 3000,
                "total_fwd_packets": 480,
                "total_backward_packets": 1,
                "total_length_of_fwd_packets": 28800,
                "total_length_of_bwd_packets": 0,
                "flow_bytes_s": 9600000.0,
                "flow_packets_s": 160000.0,
                "fwd_packet_length_mean": 60.0,
                "bwd_packet_length_mean": 0.0,
                "packet_length_mean": 60.0,
                "packet_length_std": 2.0,
                "syn_flag_count": 480,
                "ack_flag_count": 0,
                "fin_flag_count": 0,
                "psh_flag_count": 0,
            }
        },
    )

    destination_port: int = Field(..., ge=0, le=65535, description="TCP/UDP destination port.")
    protocol: int | None = Field(None, ge=0, le=255, description="IANA protocol number (6=TCP, 17=UDP).")

    flow_duration: float | None = Field(None, ge=0, le=_MAX_RATE, description="Flow duration in microseconds.")
    total_fwd_packets: float | None = Field(None, ge=0, le=_MAX_COUNT)
    total_backward_packets: float | None = Field(None, ge=0, le=_MAX_COUNT)
    total_length_of_fwd_packets: float | None = Field(None, ge=0, le=_MAX_BYTES)
    total_length_of_bwd_packets: float | None = Field(None, ge=0, le=_MAX_BYTES)

    fwd_packet_length_mean: float | None = Field(None, ge=0, le=_MAX_BYTES)
    fwd_packet_length_std: float | None = Field(None, ge=0, le=_MAX_BYTES)
    bwd_packet_length_mean: float | None = Field(None, ge=0, le=_MAX_BYTES)
    bwd_packet_length_std: float | None = Field(None, ge=0, le=_MAX_BYTES)

    flow_bytes_s: float | None = Field(None, ge=0, le=_MAX_RATE, description="Bytes per second.")
    flow_packets_s: float | None = Field(None, ge=0, le=_MAX_RATE, description="Packets per second.")
    flow_iat_mean: float | None = Field(None, ge=0, le=_MAX_RATE)
    flow_iat_std: float | None = Field(None, ge=0, le=_MAX_RATE)

    min_packet_length: float | None = Field(None, ge=0, le=_MAX_BYTES)
    max_packet_length: float | None = Field(None, ge=0, le=_MAX_BYTES)
    packet_length_mean: float | None = Field(None, ge=0, le=_MAX_BYTES)
    packet_length_std: float | None = Field(None, ge=0, le=_MAX_BYTES)
    average_packet_size: float | None = Field(None, ge=0, le=_MAX_BYTES)

    syn_flag_count: float | None = Field(None, ge=0, le=_MAX_COUNT)
    ack_flag_count: float | None = Field(None, ge=0, le=_MAX_COUNT)
    fin_flag_count: float | None = Field(None, ge=0, le=_MAX_COUNT)
    psh_flag_count: float | None = Field(None, ge=0, le=_MAX_COUNT)
    rst_flag_count: float | None = Field(None, ge=0, le=_MAX_COUNT)
    urg_flag_count: float | None = Field(None, ge=0, le=_MAX_COUNT)

    down_up_ratio: float | None = Field(None, ge=0, le=_MAX_COUNT)
    # Flow meters emit -1 when no window was observed, so this one may be negative.
    init_win_bytes_forward: float | None = Field(None, ge=-1, le=_MAX_BYTES)
    init_win_bytes_backward: float | None = Field(None, ge=-1, le=_MAX_BYTES)

    @field_validator("*", mode="before")
    @classmethod
    def _reject_non_finite(cls, value: Any) -> Any:
        """Reject NaN/Infinity before they reach the pipeline.

        JSON has no NaN literal, but Python's ``json`` module happily parses
        ``NaN`` and ``Infinity``. Letting those through would silently poison
        the scaler, so they are refused at the boundary.
        """
        if isinstance(value, float):
            if value != value or value in (float("inf"), float("-inf")):
                raise ValueError("Value must be a finite number.")
        return value

    def to_flow_dict(self) -> dict[str, Any]:
        """Flatten to the plain mapping the predictor expects, dropping nulls."""
        return {key: value for key, value in self.model_dump().items() if value is not None}


class FeatureContributionResponse(BaseModel):
    """One feature's contribution to a prediction."""

    feature: str
    display_name: str
    contribution: float
    direction: str
    value: float | str | None = None


class PredictionResponse(BaseModel):
    """Result of scoring a single flow."""

    model_config = ConfigDict(protected_namespaces=())

    prediction: str = Field(..., description="Predicted attack family, or BENIGN.")
    confidence: float = Field(..., ge=0, le=1, description="Probability of the winning class.")
    risk_score: int = Field(..., ge=0, le=100, description="Composite model-derived risk score.")
    risk_level: Literal["LOW", "MEDIUM", "HIGH", "CRITICAL"]
    is_anomaly: bool = Field(..., description="Isolation Forest verdict, independent of the classifier.")
    anomaly_score: float = Field(..., ge=0, le=1, description="Normalised anomaly score; not a probability.")
    is_malicious: bool
    class_probabilities: dict[str, float]
    risk_components: dict[str, float] = Field(
        default_factory=dict, description="Per-signal breakdown of the risk score."
    )
    indicators: list[str] = Field(default_factory=list, description="Human-readable traffic red flags.")
    top_features: list[FeatureContributionResponse] = Field(default_factory=list)


class BatchSummary(BaseModel):
    """Aggregate statistics over a scored batch."""

    total_records: int
    benign_records: int = 0
    malicious_records: int = 0
    attack_rate: float = 0.0
    anomalies_detected: int = 0
    critical_threats: int = 0
    high_threats: int = 0
    mean_risk_score: float = 0.0
    max_risk_score: int = 0
    mean_confidence: float = 0.0
    attack_distribution: dict[str, int] = Field(default_factory=dict)
    risk_distribution: dict[str, int] = Field(default_factory=dict)


class BatchPredictionResponse(BaseModel):
    """Response for a CSV batch upload."""

    filename: str
    rows_received: int
    rows_scored: int
    summary: BatchSummary
    predictions: list[dict[str, Any]] = Field(
        default_factory=list,
        description="Per-row results. Truncated to the requested preview limit.",
    )
    truncated: bool = False


class ModelInfoResponse(BaseModel):
    """Metadata describing the deployed model."""

    model_config = ConfigDict(protected_namespaces=())

    model_name: str
    model_version: str
    trained_at: str
    training_rows: int
    supported_classes: list[str]
    feature_count: int
    raw_features: list[str] = Field(default_factory=list)
    transformed_feature_count: int = 0
    anomaly_detector_loaded: bool = False
    explanations_enabled: bool = False
    environment: dict[str, str] = Field(default_factory=dict)
    metrics: dict[str, Any] = Field(default_factory=dict)


class HealthResponse(BaseModel):
    """Liveness/readiness payload."""

    status: Literal["healthy", "degraded"]
    model_loaded: bool
    anomaly_detector_loaded: bool
    api_version: str
    uptime_seconds: float


class ErrorResponse(BaseModel):
    """Uniform error envelope."""

    detail: str
    error_type: str = "error"
