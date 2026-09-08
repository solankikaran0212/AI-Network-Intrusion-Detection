"""Tests for column normalisation, cleaning, feature engineering and the pipeline."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.config import CORE_FEATURES
from src.data_loader import normalize_column, normalize_columns
from src.feature_engineering import DERIVED_FEATURES, add_derived_features, bucket_port
from src.preprocessing import (
    align_features,
    build_preprocessor,
    clean_dataset,
    normalize_label,
    prepare_for_inference,
    select_feature_columns,
)


class TestColumnNormalisation:
    @pytest.mark.parametrize(
        ("raw", "expected"),
        [
            ("Flow Bytes/s", "flow_bytes_s"),
            (" Destination Port", "destination_port"),
            ("Total Length of Fwd Packets", "total_length_of_fwd_packets"),
            ("Init_Win_bytes_forward", "init_win_bytes_forward"),
            ("Down/Up Ratio", "down_up_ratio"),
        ],
    )
    def test_normalize_column(self, raw: str, expected: str) -> None:
        assert normalize_column(raw) == expected

    def test_normalize_columns_is_idempotent(self) -> None:
        frame = pd.DataFrame({"Flow Bytes/s": [1.0], " Destination Port ": [80]})
        once = normalize_columns(frame)
        twice = normalize_columns(once)
        assert list(once.columns) == list(twice.columns)


class TestLabelNormalisation:
    @pytest.mark.parametrize(
        ("raw", "expected"),
        [
            ("BENIGN", "BENIGN"),
            ("DDoS", "DDoS"),
            ("DoS Hulk", "DoS"),
            ("DoS GoldenEye", "DoS"),
            ("PortScan", "PortScan"),
            ("FTP-Patator", "BruteForce"),
            ("SSH-Patator", "BruteForce"),
            ("Bot", "Botnet"),
            ("Infiltration", "Infiltration"),
        ],
    )
    def test_known_labels_map_to_families(self, raw: str, expected: str) -> None:
        assert normalize_label(raw) == expected

    def test_mapping_is_idempotent(self) -> None:
        """A cleaned file must survive a second cleaning pass unchanged."""
        for family in ("BENIGN", "DDoS", "DoS", "PortScan", "BruteForce", "WebAttack", "Botnet"):
            assert normalize_label(normalize_label(family)) == family

    def test_missing_label_returns_empty(self) -> None:
        assert normalize_label(np.nan) == ""
        assert normalize_label("") == ""

    def test_unknown_label_is_flagged_not_dropped(self) -> None:
        assert normalize_label("SomeNovelAttack") == "Other"


class TestCleanDataset:
    def test_removes_duplicates(self, raw_frame: pd.DataFrame) -> None:
        doubled = pd.concat([raw_frame, raw_frame.head(50)], ignore_index=True)
        cleaned, report = clean_dataset(normalize_columns(doubled), min_samples_per_class=10)
        assert report.duplicates_removed >= 50
        assert len(cleaned) < len(doubled)

    def test_replaces_infinities(self, raw_frame: pd.DataFrame) -> None:
        frame = normalize_columns(raw_frame.copy())
        frame.loc[frame.index[:20], "flow_bytes_s"] = np.inf
        frame.loc[frame.index[20:40], "flow_packets_s"] = -np.inf
        cleaned, report = clean_dataset(frame, min_samples_per_class=10)
        assert report.infinite_values_replaced >= 40
        numeric = cleaned.select_dtypes(include=[np.number])
        assert not np.isinf(numeric.to_numpy()).any()

    def test_drops_leakage_columns(self, raw_frame: pd.DataFrame) -> None:
        frame = normalize_columns(raw_frame.copy())
        frame["source_ip"] = "10.0.0.1"
        frame["destination_ip"] = "10.0.0.2"
        frame["flow_id"] = "abc"
        cleaned, _ = clean_dataset(frame, min_samples_per_class=10)
        for column in ("source_ip", "destination_ip", "flow_id"):
            assert column not in cleaned.columns

    def test_produces_label_column(self, cleaned_frame: pd.DataFrame) -> None:
        assert "label" in cleaned_frame.columns
        assert cleaned_frame["label"].notna().all()
        assert "Other" not in set(cleaned_frame["label"])


class TestFeatureEngineering:
    def test_all_derived_features_created(self, cleaned_frame: pd.DataFrame) -> None:
        enriched = add_derived_features(cleaned_frame)
        for feature in DERIVED_FEATURES:
            assert feature in enriched.columns, f"missing derived feature {feature}"

    def test_zero_duration_does_not_produce_infinity(self) -> None:
        """SYN floods routinely report a zero-microsecond duration."""
        frame = pd.DataFrame({
            "flow_duration": [0.0],
            "total_fwd_packets": [500.0],
            "total_backward_packets": [0.0],
            "total_length_of_fwd_packets": [30000.0],
            "total_length_of_bwd_packets": [0.0],
            "destination_port": [80],
        })
        enriched = add_derived_features(frame)
        assert np.isfinite(enriched["packets_per_second"]).all()
        assert np.isfinite(enriched["bytes_per_second"]).all()

    def test_zero_backward_packets_is_safe(self) -> None:
        """A scan receives no replies; the ratio must not divide by zero."""
        frame = pd.DataFrame({
            "total_fwd_packets": [10.0], "total_backward_packets": [0.0],
            "total_length_of_fwd_packets": [0.0], "total_length_of_bwd_packets": [0.0],
            "flow_duration": [100.0], "destination_port": [22],
        })
        enriched = add_derived_features(frame)
        assert np.isfinite(enriched["fwd_bwd_packet_ratio"]).all()

    def test_does_not_mutate_input(self, cleaned_frame: pd.DataFrame) -> None:
        before = list(cleaned_frame.columns)
        add_derived_features(cleaned_frame)
        assert list(cleaned_frame.columns) == before

    @pytest.mark.parametrize(
        ("port", "expected"),
        [
            (80, "http"),
            (443, "https"),
            (22, "ssh"),
            (53, "dns"),
            (3389, "remote_desktop"),
            (3306, "database"),
            (55000, "ephemeral"),
        ],
    )
    def test_port_bucketing(self, port: int, expected: str) -> None:
        assert bucket_port(pd.Series([port])).iloc[0] == expected


class TestPreprocessorPipeline:
    def test_selects_core_features(self, cleaned_frame: pd.DataFrame) -> None:
        selected = select_feature_columns(cleaned_frame.drop(columns=["label"]), restrict_to_core=True)
        assert selected
        assert set(selected).issubset(set(CORE_FEATURES))

    def test_fit_transform_shape_is_stable(self, cleaned_frame: pd.DataFrame) -> None:
        features = select_feature_columns(cleaned_frame.drop(columns=["label"]))
        pipeline = build_preprocessor(features)
        matrix = pipeline.fit_transform(cleaned_frame[features])
        assert matrix.shape[0] == len(cleaned_frame)
        assert np.isfinite(matrix).all(), "pipeline emitted non-finite values"

    def test_transform_matches_training_width(self, cleaned_frame: pd.DataFrame) -> None:
        """Train/serve consistency: one row must produce the same column count."""
        features = select_feature_columns(cleaned_frame.drop(columns=["label"]))
        pipeline = build_preprocessor(features)
        train_matrix = pipeline.fit_transform(cleaned_frame[features])
        single = pipeline.transform(cleaned_frame[features].head(1))
        assert single.shape[1] == train_matrix.shape[1]

    def test_align_features_fills_missing_and_drops_extra(self) -> None:
        frame = pd.DataFrame({"destination_port": [80], "irrelevant": [1]})
        aligned = align_features(frame, ["destination_port", "flow_duration"])
        assert list(aligned.columns) == ["destination_port", "flow_duration"]
        assert aligned["flow_duration"].isna().all()

    def test_prepare_for_inference_preserves_row_count(self, cleaned_frame: pd.DataFrame) -> None:
        """Inference must never drop or deduplicate rows."""
        features = select_feature_columns(cleaned_frame.drop(columns=["label"]))
        duplicated = pd.concat([cleaned_frame.head(5)] * 3, ignore_index=True)
        prepared = prepare_for_inference(duplicated, features)
        assert len(prepared) == 15

    def test_unseen_category_does_not_crash(self, cleaned_frame: pd.DataFrame) -> None:
        """handle_unknown='ignore' must absorb a protocol never seen in training."""
        features = select_feature_columns(cleaned_frame.drop(columns=["label"]))
        pipeline = build_preprocessor(features)
        pipeline.fit(cleaned_frame[features])
        novel = cleaned_frame[features].head(3).copy()
        if "protocol" in novel.columns:
            novel["protocol"] = 47  # GRE, absent from the fixture
        assert pipeline.transform(novel).shape[0] == 3
