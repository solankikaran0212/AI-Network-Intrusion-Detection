"""Domain-driven feature engineering for network flows.

Design principle: every derived feature must correspond to something an analyst
would actually reason about when triaging traffic. Automatically generating
hundreds of polynomial interactions would inflate the feature count without
adding signal, slow SHAP to a crawl, and make the model impossible to defend in
a review. The eight features below were each chosen for a specific attack
behaviour, documented inline.

The transformation is implemented as a scikit-learn ``FunctionTransformer`` so
it lives *inside* the fitted pipeline. This is the single most important
guarantee in the project: training and inference cannot drift apart, because
they execute literally the same object.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.preprocessing import FunctionTransformer

from src.config import get_logger

logger = get_logger(__name__)

#: Microseconds. CICFlowMeter reports ``flow_duration`` in microseconds.
_MICROSECONDS_PER_SECOND = 1_000_000.0

#: Small constant added to denominators. Chosen well below any real measured
#: value so it cannot shift a legitimate ratio, but large enough to stop
#: zero-duration flows (very common in SYN floods) producing infinities.
_EPS = 1e-6

DERIVED_FEATURES: tuple[str, ...] = (
    "duration_seconds",
    "packets_per_second",
    "bytes_per_second",
    "fwd_bwd_packet_ratio",
    "fwd_bwd_byte_ratio",
    "avg_packet_size_derived",
    "tcp_flag_density",
    "bytes_per_packet",
    "is_ephemeral_port",
)


def _col(frame: pd.DataFrame, name: str, default: float = 0.0) -> pd.Series:
    """Return a numeric column, or a constant series if the column is absent.

    Real deployments receive partial feature sets - a lightweight collector may
    not emit every CICFlowMeter field. Rather than raising, we substitute a
    neutral default so a partially-specified flow still scores.
    """
    if name in frame.columns:
        return pd.to_numeric(frame[name], errors="coerce").fillna(default)
    return pd.Series(default, index=frame.index, dtype="float64")


def bucket_port(port: pd.Series) -> pd.Series:
    """Map destination port numbers to service categories.

    Port numbers are *nominal*, not ordinal: port 443 is not "more" than port
    80, and the numeric gap between 22 and 23 carries no magnitude information.
    Feeding the raw integer to a linear model is therefore meaningless, and even
    tree models waste splits discovering the buckets. We encode the service
    semantics directly instead, which also generalises: an SSH brute-force on a
    non-standard port still lands in a sensible bucket.
    """
    numeric = pd.to_numeric(port, errors="coerce").fillna(-1)
    conditions = [
        numeric.isin([80, 8080, 8000, 8008]),
        numeric.isin([443, 8443]),
        numeric.isin([22]),
        numeric.isin([21, 20]),
        numeric.isin([23]),
        numeric.isin([25, 465, 587, 110, 143]),
        numeric.isin([53]),
        numeric.isin([3389, 5900]),
        numeric.isin([445, 139, 135]),
        numeric.isin([1433, 3306, 5432, 27017, 6379]),
        (numeric >= 0) & (numeric <= 1023),
        (numeric >= 1024) & (numeric <= 49151),
        numeric >= 49152,
    ]
    choices = [
        "http",
        "https",
        "ssh",
        "ftp",
        "telnet",
        "mail",
        "dns",
        "remote_desktop",
        "smb",
        "database",
        "other_well_known",
        "registered",
        "ephemeral",
    ]
    return pd.Series(np.select(conditions, choices, default="unknown"), index=port.index, dtype="object")


def add_derived_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Append engineered features to a flow frame.

    Each feature and the attack behaviour it targets:

    ``packets_per_second`` / ``bytes_per_second``
        Volumetric rate. This is the primary discriminator for DoS and DDoS: a
        flood is defined by rate, not by total volume, so a raw packet count
        cannot separate a flood from a long-running legitimate download.

    ``fwd_bwd_packet_ratio``
        Directional asymmetry. A normal TCP conversation is roughly balanced.
        A port scan sends many SYNs and receives almost nothing, producing an
        extreme ratio; this is the single cleanest scan indicator.

    ``fwd_bwd_byte_ratio``
        Separates "many small requests" (scanning, brute force) from
        "small request, huge response" (data exfiltration, DNS amplification).

    ``avg_packet_size_derived`` / ``bytes_per_packet``
        Payload shape. Scans and floods use minimal or empty packets near the
        MTU floor; real sessions show a mixture of small ACKs and full-MTU data
        segments.

    ``tcp_flag_density``
        Control-flag count normalised by packet count. A well-behaved flow uses
        a handful of flags across its lifetime. SYN floods and stealth scans
        (FIN/NULL/XMAS) push this ratio towards 1, because nearly every packet
        is a control packet with no data.

    ``is_ephemeral_port``
        Whether the destination is a high, unregistered port - weak on its own,
        but useful in combination with botnet C2 traffic which frequently avoids
        well-known service ports.

    Args:
        frame: Flow records with normalized column names.

    Returns:
        A new frame with the derived columns appended. The input is not mutated.
    """
    out = frame.copy()

    duration_us = _col(out, "flow_duration")
    duration_s = duration_us.abs() / _MICROSECONDS_PER_SECOND
    out["duration_seconds"] = duration_s

    fwd_pkts = _col(out, "total_fwd_packets")
    bwd_pkts = _col(out, "total_backward_packets")
    fwd_bytes = _col(out, "total_length_of_fwd_packets")
    bwd_bytes = _col(out, "total_length_of_bwd_packets")

    total_pkts = fwd_pkts + bwd_pkts
    total_bytes = fwd_bytes + bwd_bytes

    out["packets_per_second"] = total_pkts / (duration_s + _EPS)
    out["bytes_per_second"] = total_bytes / (duration_s + _EPS)
    out["fwd_bwd_packet_ratio"] = fwd_pkts / (bwd_pkts + 1.0)
    out["fwd_bwd_byte_ratio"] = fwd_bytes / (bwd_bytes + 1.0)
    out["avg_packet_size_derived"] = total_bytes / (total_pkts + _EPS)
    out["bytes_per_packet"] = out["avg_packet_size_derived"]

    flags = (
        _col(out, "syn_flag_count")
        + _col(out, "ack_flag_count")
        + _col(out, "fin_flag_count")
        + _col(out, "psh_flag_count")
        + _col(out, "rst_flag_count")
        + _col(out, "urg_flag_count")
    )
    out["tcp_flag_density"] = flags / (total_pkts + _EPS)

    port = _col(out, "destination_port", default=-1.0)
    out["is_ephemeral_port"] = (port >= 49152).astype("int8")
    out["port_category"] = bucket_port(port)

    if "protocol" in out.columns:
        out["protocol"] = _protocol_names(out["protocol"])
    else:
        out["protocol"] = "unknown"

    # Rates are unbounded by construction; clip the extreme tail so a single
    # zero-duration flow cannot dominate the scaler's variance estimate.
    for column in ("packets_per_second", "bytes_per_second", "fwd_bwd_packet_ratio", "fwd_bwd_byte_ratio"):
        out[column] = out[column].replace([np.inf, -np.inf], np.nan).clip(upper=1e9)

    return out


def _protocol_names(series: pd.Series) -> pd.Series:
    """Map IANA protocol numbers to names, passing through existing strings."""
    mapping = {0: "hopopt", 1: "icmp", 6: "tcp", 17: "udp", 47: "gre", 58: "icmpv6"}
    numeric = pd.to_numeric(series, errors="coerce")
    if numeric.notna().mean() > 0.5:
        return numeric.map(mapping).fillna("other").astype("object")
    return series.astype("object").fillna("unknown")


#: Pipeline-embeddable version of :func:`add_derived_features`.
#: ``validate=False`` keeps the DataFrame (and therefore column names) intact,
#: which the downstream ColumnTransformer needs.
feature_engineering_transformer = FunctionTransformer(
    add_derived_features, validate=False, feature_names_out=None
)


def engineered_feature_names(base_features: list[str]) -> list[str]:
    """Return the full post-engineering feature list, for documentation/tests."""
    return list(dict.fromkeys([*base_features, *DERIVED_FEATURES, "port_category", "protocol"]))
