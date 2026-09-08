"""Loading and merging of raw CIC-IDS2017 capture files.

The eight CSVs that make up CIC-IDS2017 were exported by CICFlowMeter on
different days and are *not* byte-identical in schema: column names carry
leading spaces, one file writes ``Fwd Header Length`` twice, and the label
column is sometimes ``Label`` and sometimes `` Label``. This module absorbs all
of that so the rest of the pipeline can assume one clean, snake_cased frame.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Iterable, Iterator

import numpy as np
import pandas as pd

from src.config import PATHS, get_logger

logger = get_logger(__name__)

_NON_ALNUM = re.compile(r"[^0-9a-zA-Z]+")


def normalize_column(name: str) -> str:
    """Normalize one raw column header to a stable snake_case identifier.

    ``' Flow Bytes/s'`` -> ``'flow_bytes_s'``; ``'Fwd IAT Total'`` ->
    ``'fwd_iat_total'``. Runs of non-alphanumeric characters collapse to a
    single underscore, which makes the function idempotent.
    """
    cleaned = _NON_ALNUM.sub("_", str(name).strip()).strip("_").lower()
    return cleaned or "unnamed"


def normalize_columns(frame: pd.DataFrame) -> pd.DataFrame:
    """Apply :func:`normalize_column` to every header, de-duplicating collisions.

    CIC-IDS2017 genuinely contains two ``Fwd Header Length`` columns. Rather than
    silently dropping one, we suffix repeats (``_1``, ``_2``) so the duplicate is
    visible and can be removed explicitly by the preprocessing stage.
    """
    seen: dict[str, int] = {}
    columns: list[str] = []
    for raw in frame.columns:
        base = normalize_column(raw)
        if base in seen:
            seen[base] += 1
            columns.append(f"{base}_{seen[base]}")
        else:
            seen[base] = 0
            columns.append(base)
    out = frame.copy()
    out.columns = columns
    return out


def discover_csv_files(directory: Path | None = None) -> list[Path]:
    """Return every CSV under ``directory`` (default: configured raw data dir)."""
    target = Path(directory) if directory is not None else PATHS.raw_data
    if not target.exists():
        logger.warning("Raw data directory does not exist: %s", target)
        return []
    files = sorted(p for p in target.rglob("*.csv") if p.is_file())
    logger.info("Discovered %d CSV file(s) in %s", len(files), target)
    return files


def _read_one(path: Path, chunksize: int | None) -> Iterator[pd.DataFrame]:
    """Read a single CSV defensively, yielding one or more chunks."""
    read_kwargs = {
        "low_memory": False,
        "encoding": "utf-8",
        "encoding_errors": "replace",
        "skipinitialspace": True,
        "on_bad_lines": "warn",
    }
    if chunksize:
        yield from pd.read_csv(path, chunksize=chunksize, **read_kwargs)
    else:
        yield pd.read_csv(path, **read_kwargs)


def load_raw_data(
    directory: Path | None = None,
    files: Iterable[Path] | None = None,
    chunksize: int | None = None,
    nrows_per_file: int | None = None,
) -> pd.DataFrame:
    """Load and vertically concatenate the raw capture files.

    Args:
        directory: Folder to scan. Defaults to the configured raw data path.
        files: Explicit file list; overrides ``directory`` when supplied.
        chunksize: Read each file in chunks of this many rows. Useful on
            memory-constrained machines - CIC-IDS2017 is ~2.8M rows.
        nrows_per_file: Keep at most this many rows per file, for quick smoke
            runs of the pipeline.

    Returns:
        A single frame with normalized column names and a ``source_file``
        column recording provenance.

    Raises:
        FileNotFoundError: If no CSV files were found.
    """
    paths = list(files) if files is not None else discover_csv_files(directory)
    if not paths:
        raise FileNotFoundError(
            f"No CSV files found in {directory or PATHS.raw_data}. "
            "Download CIC-IDS2017 and place the CSVs there (see README > Dataset Setup)."
        )

    frames: list[pd.DataFrame] = []
    for path in paths:
        try:
            collected: list[pd.DataFrame] = []
            rows = 0
            for chunk in _read_one(path, chunksize):
                chunk = normalize_columns(chunk)
                if nrows_per_file is not None:
                    remaining = nrows_per_file - rows
                    if remaining <= 0:
                        break
                    chunk = chunk.head(remaining)
                rows += len(chunk)
                collected.append(chunk)
            if not collected:
                continue
            frame = pd.concat(collected, ignore_index=True) if len(collected) > 1 else collected[0]
            frame["source_file"] = path.name
            frames.append(frame)
            logger.info("Loaded %-45s rows=%7d cols=%3d", path.name, len(frame), frame.shape[1])
        except Exception:  # noqa: BLE001 - one bad file must not abort the run
            logger.exception("Failed to read %s; skipping it", path.name)

    if not frames:
        raise FileNotFoundError("Every candidate CSV failed to load; see logs for details.")

    combined = pd.concat(frames, ignore_index=True, sort=False)
    logger.info("Combined dataset: %d rows x %d columns", len(combined), combined.shape[1])
    return combined


def find_label_column(frame: pd.DataFrame) -> str:
    """Locate the ground-truth label column after normalization."""
    for candidate in ("label", "labels", "attack", "class", "attack_category"):
        if candidate in frame.columns:
            return candidate
    raise KeyError(
        f"No label column found. Available columns: {sorted(frame.columns)[:20]}..."
    )


def coerce_numeric(frame: pd.DataFrame, exclude: Iterable[str] = ()) -> pd.DataFrame:
    """Force object-typed feature columns to numeric where that is meaningful.

    CICFlowMeter writes ``Infinity`` and ``NaN`` as literal strings in the
    rate columns whenever a flow lasted zero microseconds, which makes pandas
    infer ``object`` dtype for otherwise-numeric fields. We convert with
    ``errors='coerce'`` so unparseable entries become ``NaN`` and are handled by
    the single missing-value policy in :mod:`src.preprocessing`.
    """
    excluded = set(exclude)
    out = frame.copy()
    converted = []
    for column in out.columns:
        if column in excluded or out[column].dtype != object:
            continue
        numeric = pd.to_numeric(out[column], errors="coerce")
        # Only accept the conversion if it did not destroy most of the column.
        if numeric.notna().mean() >= 0.90:
            out[column] = numeric
            converted.append(column)
    if converted:
        logger.info("Coerced %d object column(s) to numeric", len(converted))
    return out.replace([np.inf, -np.inf], np.nan)


def load_sample_data(path: Path | None = None) -> pd.DataFrame:
    """Load the small bundled sample used by tests, demos and the simulator."""
    target = Path(path) if path is not None else PATHS.sample_flows
    if not target.exists():
        raise FileNotFoundError(
            f"Sample file not found at {target}. Run `python scripts/generate_sample_data.py` first."
        )
    frame = normalize_columns(pd.read_csv(target))
    logger.info("Loaded sample data: %d rows x %d columns", len(frame), frame.shape[1])
    return frame
