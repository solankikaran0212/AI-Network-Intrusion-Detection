#!/usr/bin/env python3
"""Generate a small SYNTHETIC sample dataset in the CIC-IDS2017 schema.

.. warning::

   **This is not real network traffic.** Every row is drawn from a parametric
   distribution written by hand to approximate the *shape* of CIC-IDS2017
   flows. It exists so that:

   * the repository can be cloned and run end-to-end in under a minute,
   * the unit tests have deterministic fixtures,
   * the dashboard's live-monitor simulation has something to stream,

   without shipping a 1.5 GB dataset or asking a reviewer to register for a
   download.

   Any metric computed on this data measures whether the *pipeline* works. It
   says nothing about real-world detection performance, and must never be
   reported as if it did. To get defensible numbers, download the real
   CIC-IDS2017 CSVs into ``data/raw/`` and retrain - see README > Dataset Setup.

The per-class parameters below encode the documented behaviour of each attack
(flood = high rate + tiny packets + SYN-dominated; scan = no response traffic;
brute force = many short repeated sessions to an auth port; and so on). They are
plausible, not measured.
"""

from __future__ import annotations

import argparse
import sys
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import PATHS, get_logger  # noqa: E402

logger = get_logger("generate_sample_data")

RNG_SEED = 42

# Per-class generative parameters. Tuples are (mean, std) for a lognormal-ish
# draw unless noted. Ports are sampled from the listed pool.
CLASS_PROFILES: dict[str, dict] = {
    "BENIGN": {
        "weight": 0.55,
        "ports": [80, 443, 443, 443, 53, 22, 25, 8080, 3306, 993],
        "duration_us": (4_000_000, 3_000_000),
        "fwd_packets": (18, 12),
        "bwd_packets": (16, 11),
        "fwd_pkt_len": (180, 120),
        "bwd_pkt_len": (700, 450),
        "syn": (1, 1),
        "ack": (14, 9),
        "fin": (1, 1),
        "psh": (5, 4),
        "rst": (0, 1),
    },
    "DDoS": {
        "weight": 0.10,
        "ports": [80, 80, 443, 8080],
        "duration_us": (2_500, 2_000),
        "fwd_packets": (420, 200),
        "bwd_packets": (2, 2),
        "fwd_pkt_len": (62, 8),
        "bwd_pkt_len": (0, 1),
        "syn": (400, 190),
        "ack": (3, 3),
        "fin": (0, 1),
        "psh": (0, 1),
        "rst": (2, 2),
    },
    "DoS": {
        "weight": 0.10,
        "ports": [80, 443, 8080],
        "duration_us": (900_000, 700_000),
        "fwd_packets": (95, 55),
        "bwd_packets": (6, 5),
        "fwd_pkt_len": (240, 90),
        "bwd_pkt_len": (120, 90),
        "syn": (3, 2),
        "ack": (70, 40),
        "fin": (0, 1),
        "psh": (60, 35),
        "rst": (1, 1),
    },
    "PortScan": {
        "weight": 0.10,
        "ports": [21, 22, 23, 25, 53, 135, 139, 445, 1433, 3306, 3389, 5900, 8080, 9200],
        "duration_us": (150, 200),
        "fwd_packets": (2, 1),
        "bwd_packets": (0, 1),
        "fwd_pkt_len": (0, 1),
        "bwd_pkt_len": (0, 1),
        "syn": (2, 1),
        "ack": (0, 1),
        "fin": (0, 1),
        "psh": (0, 1),
        "rst": (1, 1),
    },
    "BruteForce": {
        "weight": 0.06,
        "ports": [22, 21, 3389, 80],
        "duration_us": (350_000, 250_000),
        "fwd_packets": (14, 7),
        "bwd_packets": (12, 6),
        "fwd_pkt_len": (95, 40),
        "bwd_pkt_len": (110, 55),
        "syn": (2, 1),
        "ack": (11, 6),
        "fin": (2, 1),
        "psh": (7, 4),
        "rst": (1, 1),
    },
    "WebAttack": {
        "weight": 0.04,
        "ports": [80, 443, 8080],
        "duration_us": (1_100_000, 800_000),
        "fwd_packets": (11, 6),
        "bwd_packets": (10, 5),
        "fwd_pkt_len": (620, 320),
        "bwd_pkt_len": (900, 600),
        "syn": (1, 1),
        "ack": (9, 5),
        "fin": (1, 1),
        "psh": (6, 3),
        "rst": (0, 1),
    },
    "Botnet": {
        "weight": 0.03,
        "ports": [8080, 6667, 443, 53, 49512, 51234],
        "duration_us": (15_000_000, 9_000_000),
        "fwd_packets": (7, 4),
        "bwd_packets": (6, 4),
        "fwd_pkt_len": (130, 60),
        "bwd_pkt_len": (150, 80),
        "syn": (1, 1),
        "ack": (6, 3),
        "fin": (1, 1),
        "psh": (3, 2),
        "rst": (0, 1),
    },
    "Infiltration": {
        "weight": 0.02,
        "ports": [445, 139, 4444, 8443, 44818],
        "duration_us": (25_000_000, 15_000_000),
        "fwd_packets": (30, 20),
        "bwd_packets": (140, 90),
        "fwd_pkt_len": (110, 70),
        "bwd_pkt_len": (1_250, 300),
        "syn": (1, 1),
        "ack": (150, 80),
        "fin": (1, 1),
        "psh": (100, 60),
        "rst": (0, 1),
    },
}


def _positive_normal(rng: np.random.Generator, mean: float, std: float, size: int, minimum: float = 0.0) -> np.ndarray:
    """Draw a clipped normal - keeps values physically plausible (non-negative)."""
    return np.clip(rng.normal(mean, max(std, 1e-6), size), minimum, None)


def _generate_class(label: str, profile: dict, count: int, rng: np.random.Generator, start: datetime) -> pd.DataFrame:
    """Generate ``count`` synthetic flows for one class."""
    duration = _positive_normal(rng, *profile["duration_us"], count, minimum=1.0)
    fwd_packets = np.round(_positive_normal(rng, *profile["fwd_packets"], count, minimum=1.0))
    bwd_packets = np.round(_positive_normal(rng, *profile["bwd_packets"], count, minimum=0.0))
    fwd_len_mean = _positive_normal(rng, *profile["fwd_pkt_len"], count)
    bwd_len_mean = _positive_normal(rng, *profile["bwd_pkt_len"], count)

    fwd_bytes = fwd_packets * fwd_len_mean
    bwd_bytes = bwd_packets * bwd_len_mean
    total_packets = fwd_packets + bwd_packets
    total_bytes = fwd_bytes + bwd_bytes
    duration_s = duration / 1_000_000.0

    packet_len_mean = total_bytes / np.maximum(total_packets, 1)
    packet_len_std = np.abs(fwd_len_mean - bwd_len_mean) / 2.0 + rng.uniform(0, 25, count)

    timestamps = [start + timedelta(seconds=float(s)) for s in np.sort(rng.uniform(0, 86_400, count))]

    frame = pd.DataFrame(
        {
            "Destination Port": rng.choice(profile["ports"], count),
            "Protocol": rng.choice([6, 6, 6, 17], count),
            "Timestamp": timestamps,
            "Flow Duration": duration.astype("int64"),
            "Total Fwd Packets": fwd_packets.astype("int64"),
            "Total Backward Packets": bwd_packets.astype("int64"),
            "Total Length of Fwd Packets": fwd_bytes.round().astype("int64"),
            "Total Length of Bwd Packets": bwd_bytes.round().astype("int64"),
            "Fwd Packet Length Mean": fwd_len_mean.round(2),
            "Fwd Packet Length Std": (fwd_len_mean * rng.uniform(0.1, 0.5, count)).round(2),
            "Bwd Packet Length Mean": bwd_len_mean.round(2),
            "Bwd Packet Length Std": (bwd_len_mean * rng.uniform(0.1, 0.5, count)).round(2),
            "Flow Bytes/s": (total_bytes / np.maximum(duration_s, 1e-6)).round(2),
            "Flow Packets/s": (total_packets / np.maximum(duration_s, 1e-6)).round(2),
            "Flow IAT Mean": (duration / np.maximum(total_packets, 1)).round(2),
            "Flow IAT Std": (duration / np.maximum(total_packets, 1) * rng.uniform(0.2, 1.5, count)).round(2),
            "Min Packet Length": np.minimum(fwd_len_mean, bwd_len_mean).round().astype("int64"),
            "Max Packet Length": np.maximum(fwd_len_mean, bwd_len_mean).round().astype("int64") + 40,
            "Packet Length Mean": packet_len_mean.round(2),
            "Packet Length Std": packet_len_std.round(2),
            "SYN Flag Count": np.round(_positive_normal(rng, *profile["syn"], count)).astype("int64"),
            "ACK Flag Count": np.round(_positive_normal(rng, *profile["ack"], count)).astype("int64"),
            "FIN Flag Count": np.round(_positive_normal(rng, *profile["fin"], count)).astype("int64"),
            "PSH Flag Count": np.round(_positive_normal(rng, *profile["psh"], count)).astype("int64"),
            "RST Flag Count": np.round(_positive_normal(rng, *profile["rst"], count)).astype("int64"),
            "URG Flag Count": rng.integers(0, 2, count),
            "Down/Up Ratio": np.round(bwd_packets / np.maximum(fwd_packets, 1)).astype("int64"),
            "Average Packet Size": packet_len_mean.round(2),
            "Init_Win_bytes_forward": rng.choice([-1, 512, 1024, 8192, 29200, 65535], count),
            "Init_Win_bytes_backward": rng.choice([-1, 229, 235, 26883, 65535], count),
            "Source IP": [f"192.168.{rng.integers(1, 20)}.{rng.integers(2, 254)}" for _ in range(count)],
            "Destination IP": [f"10.0.{rng.integers(0, 5)}.{rng.integers(2, 254)}" for _ in range(count)],
            "Label": label,
        }
    )
    return frame


def _inject_realistic_defects(frame: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    """Introduce the data-quality problems the real CSVs actually contain.

    Without these, the preprocessing pipeline's cleaning branches would never be
    exercised and the tests would give false confidence. We inject:

    * literal ``Infinity`` strings in rate columns (zero-duration flows),
    * a small fraction of NaNs,
    * exact duplicate rows.
    """
    out = frame.copy()

    inf_idx = rng.choice(len(out), size=max(1, len(out) // 200), replace=False)
    out.loc[inf_idx, "Flow Bytes/s"] = np.inf
    out.loc[inf_idx, "Flow Packets/s"] = np.inf

    nan_idx = rng.choice(len(out), size=max(1, len(out) // 150), replace=False)
    out.loc[nan_idx, "Flow IAT Std"] = np.nan

    dup_idx = rng.choice(len(out), size=max(1, len(out) // 100), replace=False)
    out = pd.concat([out, out.iloc[dup_idx]], ignore_index=True)

    return out.sample(frac=1.0, random_state=RNG_SEED).reset_index(drop=True)


def generate(n_rows: int = 20_000, seed: int = RNG_SEED, with_defects: bool = True) -> pd.DataFrame:
    """Generate the full synthetic dataset across all classes."""
    rng = np.random.default_rng(seed)
    start = datetime(2017, 7, 5, 8, 0, 0)

    frames = []
    for label, profile in CLASS_PROFILES.items():
        count = max(int(n_rows * profile["weight"]), 60)
        frames.append(_generate_class(label, profile, count, rng, start))
        logger.info("Generated %-14s %6d synthetic flows", label, count)

    combined = pd.concat(frames, ignore_index=True)
    if with_defects:
        combined = _inject_realistic_defects(combined, rng)
    return combined.sample(frac=1.0, random_state=seed).reset_index(drop=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--rows", type=int, default=20_000, help="Approximate total rows to generate.")
    parser.add_argument("--seed", type=int, default=RNG_SEED, help="Random seed for reproducibility.")
    parser.add_argument("--output", type=Path, default=None, help="Output CSV path.")
    parser.add_argument(
        "--to-raw",
        action="store_true",
        help="Also write a copy into data/raw/ so the full pipeline can run without the real dataset.",
    )
    args = parser.parse_args()

    PATHS.ensure()
    frame = generate(n_rows=args.rows, seed=args.seed)

    output = args.output or PATHS.sample_flows
    output.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output, index=False)
    logger.info("Wrote %d synthetic flows to %s", len(frame), output)

    if args.to_raw:
        raw_target = PATHS.raw_data / "SYNTHETIC_sample_traffic.csv"
        raw_target.parent.mkdir(parents=True, exist_ok=True)
        frame.to_csv(raw_target, index=False)
        logger.info("Wrote a copy to %s (clearly named SYNTHETIC)", raw_target)

    print("\n" + "=" * 70)
    print("SYNTHETIC DATA - NOT REAL NETWORK TRAFFIC")
    print("=" * 70)
    print(f"Rows: {len(frame):,}   Columns: {frame.shape[1]}")
    print("\nClass distribution:")
    print(frame["Label"].value_counts().to_string())
    print("\nMetrics computed on this file validate the PIPELINE only.")
    print("Download real CIC-IDS2017 CSVs into data/raw/ for meaningful results.")
    print("=" * 70)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
