#!/usr/bin/env python3
"""Stage 1 of the pipeline: turn raw CSVs into one cleaned, modelling-ready file.

Run this before ``train_model.py``. It is a separate script on purpose -
cleaning 2.8M rows takes minutes, and you will retrain far more often than you
re-clean. Caching the processed artifact keeps the training loop fast.

Usage::

    python scripts/prepare_data.py                       # read data/raw/
    python scripts/prepare_data.py --use-sample          # use the synthetic sample
    python scripts/prepare_data.py --nrows-per-file 50000  # quick smoke run
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd  # noqa: E402

from src.config import PATHS, get_logger  # noqa: E402
from src.data_loader import discover_csv_files, load_raw_data, load_sample_data  # noqa: E402
from src.preprocessing import clean_dataset  # noqa: E402

logger = get_logger("prepare_data")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--raw-dir", type=Path, default=None, help="Directory of raw CSVs (default: data/raw).")
    parser.add_argument("--output", type=Path, default=None, help="Output path (default: data/processed/dataset.parquet).")
    parser.add_argument("--nrows-per-file", type=int, default=None, help="Cap rows read per CSV, for a fast smoke run.")
    parser.add_argument("--chunksize", type=int, default=None, help="Read CSVs in chunks of this size (low-memory mode).")
    parser.add_argument("--min-per-class", type=int, default=None, help="Drop classes with fewer rows than this.")
    parser.add_argument("--use-sample", action="store_true", help="Use the synthetic sample instead of data/raw/.")
    args = parser.parse_args()

    PATHS.ensure()

    if args.use_sample:
        logger.warning("Using SYNTHETIC sample data - results validate the pipeline, not real-world performance.")
        raw = load_sample_data()
    else:
        raw_dir = args.raw_dir or PATHS.raw_data
        if not discover_csv_files(raw_dir):
            logger.error(
                "No CSV files in %s.\n"
                "  Option A: download CIC-IDS2017 CSVs there (see README > Dataset Setup)\n"
                "  Option B: run `python scripts/generate_sample_data.py` then re-run with --use-sample",
                raw_dir,
            )
            return 1
        raw = load_raw_data(directory=raw_dir, chunksize=args.chunksize, nrows_per_file=args.nrows_per_file)

    logger.info("Raw shape: %s", raw.shape)
    cleaned, report = clean_dataset(raw, min_samples_per_class=args.min_per_class)

    output = args.output or PATHS.processed_dataset
    output.parent.mkdir(parents=True, exist_ok=True)
    try:
        cleaned.to_parquet(output, index=False)
    except Exception:  # noqa: BLE001 - pyarrow may be unavailable
        output = output.with_suffix(".csv")
        cleaned.to_csv(output, index=False)
        logger.warning("Parquet unavailable; wrote CSV to %s instead.", output)
    logger.info("Wrote processed dataset to %s", output)

    report_path = PATHS.reports / "cleaning_report.json"
    with report_path.open("w", encoding="utf-8") as handle:
        json.dump(
            {
                "source": "synthetic_sample" if args.use_sample else str(args.raw_dir or PATHS.raw_data),
                "rows_in": report.rows_in,
                "rows_out": report.rows_out,
                "duplicates_removed": report.duplicates_removed,
                "null_label_rows_removed": report.null_label_rows_removed,
                "infinite_values_replaced": report.infinite_values_replaced,
                "columns_dropped": {
                    "leakage": report.leakage_columns_dropped,
                    "all_nan": report.all_nan_columns_dropped,
                    "constant": report.constant_columns_dropped,
                },
                "rare_classes_dropped": report.rare_classes_dropped,
                "class_counts": report.class_counts,
                "output_path": str(output),
            },
            handle,
            indent=2,
        )

    print("\n" + "=" * 68)
    print("DATA PREPARATION COMPLETE")
    print("=" * 68)
    print(f"  Rows       : {report.rows_in:,} -> {report.rows_out:,}")
    print(f"  Duplicates : {report.duplicates_removed:,} removed")
    print(f"  Infinities : {report.infinite_values_replaced:,} replaced with NaN")
    print(f"  Columns    : {cleaned.shape[1]} retained")
    print(f"  Output     : {output}")
    print("\n  Class distribution:")
    for label, count in sorted(report.class_counts.items(), key=lambda kv: -kv[1]):
        share = count / max(report.rows_out, 1) * 100
        print(f"    {label:<15} {count:>10,}  ({share:5.2f}%)")
    print("=" * 68)
    print("\nNext: python scripts/train_model.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
