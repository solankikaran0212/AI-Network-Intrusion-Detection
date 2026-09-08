#!/usr/bin/env python3
"""Stage 2 of the pipeline: train, compare, select and persist the models.

Reads the cleaned dataset produced by ``prepare_data.py``, trains the three
candidate classifiers, selects the winner on **validation** macro-F1, evaluates
that winner once on the held-out test split, fits the Isolation Forest, and
writes both artifacts plus a full metrics report.

Usage::

    python scripts/train_model.py                    # default run
    python scripts/train_model.py --cross-validate   # add stratified k-fold CV
    python scripts/train_model.py --all-features     # use every numeric column
    python scripts/train_model.py --input path.csv   # explicit input file
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd  # noqa: E402

from src.config import PATHS, get_logger  # noqa: E402

from src.train import run_training  # noqa: E402

logger = get_logger("train_model")

SYNTHETIC_MARKER = "SYNTHETIC"


def _load_dataset(explicit: Path | None) -> tuple[pd.DataFrame, Path]:
    """Load the processed dataset, tolerating either parquet or CSV."""
    if explicit is not None:
        path = explicit
    else:
        path = PATHS.processed_dataset
        if not path.exists():
            csv_fallback = path.with_suffix(".csv")
            path = csv_fallback if csv_fallback.exists() else path

    if not path.exists():
        raise FileNotFoundError(
            f"No processed dataset at {path}. Run `python scripts/prepare_data.py` first."
        )

    frame = pd.read_parquet(path) if path.suffix == ".parquet" else pd.read_csv(path)
    return frame, path


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--input", type=Path, default=None, help="Processed dataset path.")
    parser.add_argument(
        "--cross-validate",
        action="store_true",
        help="Run stratified k-fold CV on the training split (slower).",
    )
    parser.add_argument(
        "--all-features",
        action="store_true",
        help="Use every numeric column instead of the compact core feature set.",
    )
    parser.add_argument(
        "--no-balance",
        action="store_true",
        help="Skip majority-class downsampling of the training split.",
    )
    args = parser.parse_args()

    PATHS.ensure()
    frame, source = _load_dataset(args.input)
    logger.info("Loaded %d rows x %d columns from %s", len(frame), frame.shape[1], source)

    if "label" not in frame.columns:
        logger.error("Dataset has no 'label' column; did prepare_data.py complete?")
        return 1

    bundle, report = run_training(
        frame,
        restrict_to_core=not args.all_features,
        run_cross_validation=args.cross_validate,
    )

    table = pd.DataFrame(report["comparison_table"])
    test = report["test_result"]

    print("\n" + "=" * 78)
    print("TRAINING COMPLETE")
    print("=" * 78)
    print(f"  Source dataset : {source}")
    print(f"  Rows used      : {report['dataset']['total_rows']:,}")
    print(f"  Classes        : {', '.join(report['dataset']['classes'])}")
    print(f"  Raw features   : {report['features']['raw_count']}")
    print(f"  After encoding : {report['features']['transformed_count']}")
    print(f"  Selected model : {report['selected_model']}  "
          f"(chosen on {report['selection_split']} {report['selection_metric']})")

    print("\n  Model comparison on the VALIDATION split:")
    if not table.empty:
        print(table.to_string(index=False))

    print("\n  Held-out TEST performance of the selected model:")
    print(f"    Accuracy              : {test['accuracy']:.4f}")
    print(f"    Macro F1              : {test['macro_f1']:.4f}")
    print(f"    Weighted F1           : {test['weighted_f1']:.4f}")
    print(f"    Macro recall          : {test['macro_recall']:.4f}")
    print(f"    Attack detection rate : {test['attack_detection_rate']:.4f}")
    print(f"    False alarm rate      : {test['false_alarm_rate']:.4f}")
    print(f"    False negatives       : {test['false_negatives']:,}  "
          f"(missed attacks - the costly error)")
    print(f"    False positives       : {test['false_positives']:,}  (analyst noise)")

    print(f"\n  Model artifact  : {PATHS.model_bundle}")
    print(f"  Anomaly artifact: {PATHS.anomaly_bundle}")
    print(f"  Metrics report  : {PATHS.metrics_report}")

    if SYNTHETIC_MARKER.lower() in str(source).lower():
        print("\n  " + "!" * 70)
        print("  WARNING: trained on SYNTHETIC data. These numbers demonstrate that")
        print("  the pipeline works; they are NOT CIC-IDS2017 results.")
        print("  " + "!" * 70)

    print("=" * 78)
    print("\nNext:  uvicorn api.main:app --reload")
    print("       streamlit run dashboard/app.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
