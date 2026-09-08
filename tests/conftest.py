"""Shared pytest fixtures.

The expensive fixture here is ``trained_bundle``: it runs the real training
pipeline once per session on a small synthetic frame. Training for real (rather
than mocking the estimator) is what makes the prediction and API tests
meaningful - they exercise the same preprocessing, encoding and serialisation
path that production uses.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.train import ModelBundle, run_training  # noqa: E402


def _synthetic_frame(rows_per_class: int = 220, seed: int = 7) -> pd.DataFrame:
    """Build a small, deliberately separable multi-class flow dataset.

    Separability is intentional: these tests assert that the *plumbing* is
    correct, not that the model is accurate. A dataset the model can actually
    learn keeps the assertions stable without making them tautological.
    """
    rng = np.random.default_rng(seed)
    profiles = {
        "BENIGN": dict(port=443, dur=5e6, fwd=25, bwd=22, fbytes=3200, bbytes=18000, syn=1, ack=20),
        "DDoS": dict(port=80, dur=3e3, fwd=480, bwd=1, fbytes=28800, bbytes=0, syn=480, ack=0),
        "PortScan": dict(port=3389, dur=1.2e2, fwd=2, bwd=0, fbytes=0, bbytes=0, syn=2, ack=0),
        "BruteForce": dict(port=22, dur=9e5, fwd=40, bwd=38, fbytes=2600, bbytes=3100, syn=2, ack=36),
    }

    frames = []
    for label, profile in profiles.items():
        noise = lambda scale, size=rows_per_class: rng.normal(1.0, scale, size)  # noqa: E731
        frame = pd.DataFrame({
            "Destination Port": profile["port"],
            "Protocol": 6,
            "Flow Duration": np.abs(profile["dur"] * noise(0.08)),
            "Total Fwd Packets": np.abs(profile["fwd"] * noise(0.1)).round(),
            "Total Backward Packets": np.abs(profile["bwd"] * noise(0.1)).round(),
            "Total Length of Fwd Packets": np.abs(profile["fbytes"] * noise(0.1)),
            "Total Length of Bwd Packets": np.abs(profile["bbytes"] * noise(0.1)),
            "Fwd Packet Length Mean": np.abs(profile["fbytes"] / max(profile["fwd"], 1) * noise(0.1)),
            "Bwd Packet Length Mean": np.abs(profile["bbytes"] / max(profile["bwd"], 1) * noise(0.1)),
            "Flow Bytes/s": np.abs((profile["fbytes"] + profile["bbytes"]) / (profile["dur"] / 1e6) * noise(0.12)),
            "Flow Packets/s": np.abs((profile["fwd"] + profile["bwd"]) / (profile["dur"] / 1e6) * noise(0.12)),
            "Packet Length Mean": np.abs(profile["fbytes"] / max(profile["fwd"], 1) * noise(0.1)),
            "Packet Length Std": np.abs(60 * noise(0.2)),
            "Min Packet Length": 0,
            "Max Packet Length": np.abs(1460 * noise(0.05)),
            "SYN Flag Count": profile["syn"],
            "ACK Flag Count": profile["ack"],
            "FIN Flag Count": rng.integers(0, 2, rows_per_class),
            "PSH Flag Count": rng.integers(0, 4, rows_per_class),
            "RST Flag Count": 0,
            "URG Flag Count": 0,
            "Down/Up Ratio": 1,
            "Average Packet Size": np.abs(profile["fbytes"] / max(profile["fwd"], 1) * noise(0.1)),
            "Init_Win_bytes_forward": 8192,
            "Init_Win_bytes_backward": 8192,
            "Label": label,
        })
        frames.append(frame)

    return pd.concat(frames, ignore_index=True).sample(frac=1.0, random_state=seed).reset_index(drop=True)


@pytest.fixture(scope="session")
def raw_frame() -> pd.DataFrame:
    """Raw-style frame with original CIC-IDS2017 column casing."""
    return _synthetic_frame()


@pytest.fixture(scope="session")
def cleaned_frame(raw_frame: pd.DataFrame) -> pd.DataFrame:
    """Frame after the cleaning stage, ready for training."""
    from src.data_loader import normalize_columns
    from src.preprocessing import clean_dataset

    cleaned, _report = clean_dataset(normalize_columns(raw_frame), min_samples_per_class=10)
    return cleaned


@pytest.fixture(scope="session")
def trained_bundle(cleaned_frame: pd.DataFrame, tmp_path_factory: pytest.TempPathFactory) -> ModelBundle:
    """Train the real pipeline once, into an isolated temp directory."""
    import src.config as config_module

    workspace = tmp_path_factory.mktemp("artifacts")
    original = config_module.PATHS

    # Redirect every artifact path so the test run never touches the repo's
    # models/ or reports/ directories.
    class _TempPaths:
        root = original.root
        raw_data = workspace / "raw"
        processed_data = workspace / "processed"
        sample_data = workspace / "sample"
        models = workspace / "models"
        reports = workspace / "reports"
        logs = workspace / "logs"
        processed_dataset = workspace / "processed" / "dataset.parquet"
        model_bundle = workspace / "models" / "nids_model.joblib"
        anomaly_bundle = workspace / "models" / "anomaly_detector.joblib"
        metrics_report = workspace / "reports" / "metrics.json"
        sample_flows = workspace / "sample" / "sample_flows.csv"

        @staticmethod
        def ensure() -> None:
            for directory in (_TempPaths.raw_data, _TempPaths.processed_data, _TempPaths.sample_data,
                              _TempPaths.models, _TempPaths.reports, _TempPaths.logs):
                directory.mkdir(parents=True, exist_ok=True)

    _TempPaths.ensure()

    import src.anomaly_detection as anomaly_module
    import src.evaluate as evaluate_module
    import src.train as train_module

    for module in (config_module, train_module, anomaly_module, evaluate_module):
        module.PATHS = _TempPaths  # type: ignore[assignment]

    bundle, _report = run_training(cleaned_frame, restrict_to_core=True)
    return bundle


@pytest.fixture(scope="session")
def predictor(trained_bundle: ModelBundle):
    """A ThreatPredictor backed by the session-trained bundle."""
    from src.anomaly_detection import load_anomaly_bundle
    from src.predict import ThreatPredictor
    import src.train as train_module

    try:
        anomaly = load_anomaly_bundle(train_module.PATHS.anomaly_bundle)
    except FileNotFoundError:
        anomaly = None

    return ThreatPredictor(model_bundle=trained_bundle, anomaly_bundle=anomaly, enable_explanations=False)
