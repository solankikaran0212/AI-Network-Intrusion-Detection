"""Central configuration for the intrusion detection system.

Every path and tunable parameter is resolved here so that no other module
hardcodes a filesystem location. Values can be overridden with environment
variables (see ``.env.example``), which keeps the same codebase usable on a
laptop, in CI, and inside a Docker container without edits.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Final

# --------------------------------------------------------------------------- #
# Paths
# --------------------------------------------------------------------------- #

PROJECT_ROOT: Final[Path] = Path(__file__).resolve().parent.parent


def _env_path(var: str, default: Path) -> Path:
    """Return a path from the environment, falling back to a project default."""
    raw = os.getenv(var)
    return Path(raw).expanduser().resolve() if raw else default


@dataclass(frozen=True)
class Paths:
    """Filesystem layout of the project."""

    root: Path = PROJECT_ROOT
    raw_data: Path = field(default_factory=lambda: _env_path("NIDS_RAW_DATA_DIR", PROJECT_ROOT / "data" / "raw"))
    processed_data: Path = field(
        default_factory=lambda: _env_path("NIDS_PROCESSED_DATA_DIR", PROJECT_ROOT / "data" / "processed")
    )
    sample_data: Path = field(default_factory=lambda: _env_path("NIDS_SAMPLE_DATA_DIR", PROJECT_ROOT / "data" / "sample"))
    models: Path = field(default_factory=lambda: _env_path("NIDS_MODEL_DIR", PROJECT_ROOT / "models"))
    reports: Path = field(default_factory=lambda: _env_path("NIDS_REPORT_DIR", PROJECT_ROOT / "reports"))
    logs: Path = field(default_factory=lambda: _env_path("NIDS_LOG_DIR", PROJECT_ROOT / "logs"))

    @property
    def processed_dataset(self) -> Path:
        return self.processed_data / "dataset.parquet"

    @property
    def model_bundle(self) -> Path:
        """Single artifact holding preprocessor + classifier + metadata."""
        return self.models / "nids_model.joblib"

    @property
    def anomaly_bundle(self) -> Path:
        return self.models / "anomaly_detector.joblib"

    @property
    def metrics_report(self) -> Path:
        return self.reports / "metrics.json"

    @property
    def sample_flows(self) -> Path:
        return self.sample_data / "sample_flows.csv"

    def ensure(self) -> None:
        """Create every writable directory if it does not already exist."""
        for directory in (self.processed_data, self.models, self.reports, self.logs, self.sample_data):
            directory.mkdir(parents=True, exist_ok=True)


PATHS = Paths()


# --------------------------------------------------------------------------- #
# Label taxonomy
# --------------------------------------------------------------------------- #

BENIGN_LABEL: Final[str] = "BENIGN"

#: CIC-IDS2017 ships ~15 fine-grained labels spread across eight CSV files.
#: Several of them are near-duplicates of each other (three DoS variants, two
#: web-attack variants) and some have fewer than 20 samples, which makes them
#: statistically meaningless to learn. We therefore collapse the raw labels into
#: a coarser taxonomy that is both learnable and operationally meaningful to an
#: analyst. The mapping is applied on a normalized (lowercase, alphanumeric)
#: form of the raw label so that encoding quirks in the original CSVs
#: (non-breaking hyphens, stray whitespace) do not break it.
LABEL_MAP: Final[dict[str, str]] = {
    "benign": "BENIGN",
    "ddos": "DDoS",
    "dos hulk": "DoS",
    "dos goldeneye": "DoS",
    "dos slowloris": "DoS",
    "dos slowhttptest": "DoS",
    "heartbleed": "DoS",
    "portscan": "PortScan",
    "port scan": "PortScan",
    "ftp patator": "BruteForce",
    "ssh patator": "BruteForce",
    "web attack brute force": "BruteForce",
    "web attack xss": "WebAttack",
    "web attack sql injection": "WebAttack",
    "bot": "Botnet",
    "botnet": "Botnet",
    "infiltration": "Infiltration",
    # Identity entries, listed last so the specific keys above win the
    # substring fallback. These make the mapping *idempotent*: re-cleaning an
    # already-normalised file (or loading the synthetic sample, which is
    # written with canonical names) yields the same taxonomy instead of
    # dumping everything into "Other".
    "dos": "DoS",
    "bruteforce": "BruteForce",
    "brute force": "BruteForce",
    "webattack": "WebAttack",
    "web attack": "WebAttack",
    "port scan": "PortScan",
}

#: Canonical ordering used for reports, dashboards and the API contract.
ATTACK_CLASSES: Final[tuple[str, ...]] = (
    "BENIGN",
    "DDoS",
    "DoS",
    "PortScan",
    "BruteForce",
    "WebAttack",
    "Botnet",
    "Infiltration",
)

#: Severity weighting used by the risk-scoring module. Higher means an analyst
#: should look at it sooner. These weights are a defensible engineering
#: judgement, not an industry standard - see ``src/risk_scoring.py``.
CLASS_SEVERITY: Final[dict[str, int]] = {
    "BENIGN": 0,
    "PortScan": 45,
    "WebAttack": 70,
    "BruteForce": 72,
    "DoS": 80,
    "DDoS": 88,
    "Botnet": 90,
    "Infiltration": 95,
}


# --------------------------------------------------------------------------- #
# Columns
# --------------------------------------------------------------------------- #

#: Columns dropped before training. Flow ID / IPs / timestamps are *identifiers*,
#: not behavioural signal. Keeping them invites the model to memorise which host
#: attacked which host in this particular capture - the classic form of target
#: leakage in NIDS papers that produces 99.9% accuracy and zero generalisation.
LEAKAGE_COLUMNS: Final[tuple[str, ...]] = (
    "flow_id",
    "source_ip",
    "src_ip",
    "destination_ip",
    "dst_ip",
    "source_port",
    "src_port",
    "timestamp",
    "fwd_header_length_1",
    "unnamed_0",
    "similarhttp",
    "inbound",
)

#: The 20 flow features the API contract exposes and the dashboard's manual
#: analyzer asks for. Chosen for coverage (volume, rate, size, shape, TCP flags,
#: service) while staying small enough for a human to fill in by hand.
CORE_FEATURES: Final[tuple[str, ...]] = (
    "destination_port",
    "protocol",
    "flow_duration",
    "total_fwd_packets",
    "total_backward_packets",
    "total_length_of_fwd_packets",
    "total_length_of_bwd_packets",
    "fwd_packet_length_mean",
    "fwd_packet_length_std",
    "bwd_packet_length_mean",
    "bwd_packet_length_std",
    "flow_bytes_s",
    "flow_packets_s",
    "flow_iat_mean",
    "flow_iat_std",
    "min_packet_length",
    "max_packet_length",
    "packet_length_mean",
    "packet_length_std",
    "syn_flag_count",
    "ack_flag_count",
    "fin_flag_count",
    "psh_flag_count",
    "rst_flag_count",
    "urg_flag_count",
    "down_up_ratio",
    "average_packet_size",
    "init_win_bytes_forward",
    "init_win_bytes_backward",
)

#: Treated as categorical by the preprocessing pipeline. ``destination_port`` is
#: numeric in the raw data but semantically a *service identifier* - the
#: distance between port 22 and port 80 is meaningless, so we bucket it instead
#: of scaling it (see ``feature_engineering.bucket_port``).
CATEGORICAL_FEATURES: Final[tuple[str, ...]] = ("port_category", "protocol")


# --------------------------------------------------------------------------- #
# Model & training hyper-parameters
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class TrainingConfig:
    """Hyper-parameters and split settings for the training pipeline."""

    random_state: int = int(os.getenv("NIDS_RANDOM_STATE", "42"))
    test_size: float = float(os.getenv("NIDS_TEST_SIZE", "0.20"))
    validation_size: float = float(os.getenv("NIDS_VAL_SIZE", "0.20"))
    cv_folds: int = int(os.getenv("NIDS_CV_FOLDS", "3"))
    #: Cap per class when balancing. ``None`` disables downsampling.
    max_samples_per_class: int | None = (
        int(os.getenv("NIDS_MAX_PER_CLASS")) if os.getenv("NIDS_MAX_PER_CLASS") else 150_000
    )
    #: Classes with fewer rows than this are dropped as statistically unusable.
    min_samples_per_class: int = int(os.getenv("NIDS_MIN_PER_CLASS", "50"))
    n_jobs: int = int(os.getenv("NIDS_N_JOBS", "-1"))
    #: Proportion of training flows the Isolation Forest treats as anomalous.
    anomaly_contamination: float = float(os.getenv("NIDS_CONTAMINATION", "0.05"))
    #: Selection metric. Macro-F1 weights every attack class equally, so a rare
    #: but dangerous class cannot be ignored the way it can under accuracy.
    selection_metric: str = os.getenv("NIDS_SELECTION_METRIC", "macro_f1")


TRAINING = TrainingConfig()


# --------------------------------------------------------------------------- #
# API / dashboard settings
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class ServiceConfig:
    """Runtime settings for the FastAPI service and Streamlit dashboard."""

    api_host: str = os.getenv("NIDS_API_HOST", "0.0.0.0")
    api_port: int = int(os.getenv("NIDS_API_PORT", "8000"))
    api_url: str = os.getenv("NIDS_API_URL", "http://localhost:8000")
    model_version: str = os.getenv("NIDS_MODEL_VERSION", "1.0.0")
    log_level: str = os.getenv("NIDS_LOG_LEVEL", "INFO").upper()
    #: Hard ceiling on uploaded CSV size, enforced before the file is parsed.
    max_upload_mb: float = float(os.getenv("NIDS_MAX_UPLOAD_MB", "25"))
    #: Row cap applied after parsing, to bound memory and response time.
    max_upload_rows: int = int(os.getenv("NIDS_MAX_UPLOAD_ROWS", "50000"))
    allowed_upload_suffixes: tuple[str, ...] = (".csv",)

    @property
    def max_upload_bytes(self) -> int:
        return int(self.max_upload_mb * 1024 * 1024)


SERVICE = ServiceConfig()


# --------------------------------------------------------------------------- #
# Logging
# --------------------------------------------------------------------------- #

_LOG_FORMAT = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
_configured = False


def configure_logging(level: str | None = None, to_file: bool = True) -> None:
    """Configure root logging once, for both console and a rotating-ish file.

    Log records deliberately never include raw payload rows: an intrusion
    detection system processes traffic metadata that may be sensitive, so we log
    shapes, counts and class names rather than the flows themselves.
    """
    global _configured
    if _configured:
        return

    handlers: list[logging.Handler] = [logging.StreamHandler()]
    if to_file:
        try:
            PATHS.logs.mkdir(parents=True, exist_ok=True)
            handlers.append(logging.FileHandler(PATHS.logs / "nids.log", encoding="utf-8"))
        except OSError:  # read-only filesystem (e.g. a locked-down container)
            pass

    logging.basicConfig(
        level=getattr(logging, (level or SERVICE.log_level), logging.INFO),
        format=_LOG_FORMAT,
        handlers=handlers,
        force=True,
    )
    logging.getLogger("matplotlib").setLevel(logging.WARNING)
    _configured = True


def get_logger(name: str) -> logging.Logger:
    """Return a module logger, configuring logging on first use."""
    configure_logging()
    return logging.getLogger(name)
