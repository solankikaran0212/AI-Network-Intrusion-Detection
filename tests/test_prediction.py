"""Tests for model loading, inference, risk scoring and anomaly detection.

These run against a bundle produced by the *real* training pipeline (see
``conftest.trained_bundle``) rather than a mock, so they exercise the same
preprocessing, encoding and serialisation path the API uses in production.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.config import BENIGN_LABEL, CLASS_SEVERITY
from src.predict import PredictionResult, ThreatPredictor, example_flow
from src.risk_scoring import compute_risk, risk_level_for
from src.train import ModelBundle, load_model_bundle, save_model_bundle


class TestModelBundle:
    def test_bundle_has_required_artifacts(self, trained_bundle: ModelBundle) -> None:
        assert trained_bundle.preprocessor is not None
        assert trained_bundle.model is not None
        assert trained_bundle.label_encoder is not None
        assert trained_bundle.feature_columns
        assert trained_bundle.class_names

    def test_metadata_is_serialisable(self, trained_bundle: ModelBundle) -> None:
        import json

        metadata = trained_bundle.to_metadata()
        json.dumps(metadata, default=str)  # must not raise
        assert metadata["feature_count"] == len(trained_bundle.feature_columns)
        assert metadata["model_name"] in {"LogisticRegression", "RandomForest", "XGBoost"}

    def test_roundtrip_through_disk_preserves_predictions(
        self, trained_bundle: ModelBundle, tmp_path
    ) -> None:
        """Serialisation must not change behaviour - the classic deployment bug."""
        path = tmp_path / "bundle.joblib"
        save_model_bundle(trained_bundle, path)
        reloaded = load_model_bundle(path)

        flow = pd.DataFrame([example_flow("flood")])
        original = ThreatPredictor(
            model_bundle=trained_bundle, anomaly_bundle=None, enable_explanations=False
        ).predict_frame(flow)[0]
        restored = ThreatPredictor(
            model_bundle=reloaded, anomaly_bundle=None, enable_explanations=False
        ).predict_frame(flow)[0]

        assert original.prediction == restored.prediction
        assert original.confidence == pytest.approx(restored.confidence, abs=1e-9)

    def test_missing_model_raises_file_not_found(self, tmp_path) -> None:
        with pytest.raises(FileNotFoundError):
            load_model_bundle(tmp_path / "does_not_exist.joblib")


class TestPrediction:
    def test_predict_one_returns_valid_result(self, predictor: ThreatPredictor) -> None:
        result = predictor.predict_one(example_flow("benign"), explain=False)
        assert isinstance(result, PredictionResult)
        assert result.prediction in predictor.class_names
        assert 0.0 <= result.confidence <= 1.0
        assert 0 <= result.risk_score <= 100
        assert result.risk_level in {"LOW", "MEDIUM", "HIGH", "CRITICAL"}

    def test_probabilities_sum_to_one(self, predictor: ThreatPredictor) -> None:
        result = predictor.predict_one(example_flow("scan"), explain=False)
        assert sum(result.class_probabilities.values()) == pytest.approx(1.0, abs=0.01)

    def test_confidence_matches_winning_class(self, predictor: ThreatPredictor) -> None:
        result = predictor.predict_one(example_flow("flood"), explain=False)
        assert result.class_probabilities[result.prediction] == pytest.approx(
            result.confidence, abs=0.01
        )

    def test_is_malicious_is_consistent_with_prediction(self, predictor: ThreatPredictor) -> None:
        for kind in ("benign", "scan", "flood"):
            result = predictor.predict_one(example_flow(kind), explain=False)
            assert result.is_malicious == (result.prediction != BENIGN_LABEL)

    def test_batch_returns_one_result_per_row(self, predictor: ThreatPredictor) -> None:
        frame = pd.DataFrame([example_flow("benign"), example_flow("scan"), example_flow("flood")])
        results = predictor.predict_frame(frame)
        assert len(results) == 3

    def test_batch_matches_single_prediction(self, predictor: ThreatPredictor) -> None:
        """Batching must not change any individual result."""
        flows = [example_flow("benign"), example_flow("scan"), example_flow("flood")]
        batch = predictor.predict_frame(pd.DataFrame(flows))
        for flow, batched in zip(flows, batch):
            single = predictor.predict_one(flow, explain=False)
            assert single.prediction == batched.prediction
            assert single.confidence == pytest.approx(batched.confidence, abs=1e-9)

    def test_empty_frame_returns_empty_list(self, predictor: ThreatPredictor) -> None:
        assert predictor.predict_frame(pd.DataFrame()) == []

    def test_partial_features_are_imputed(self, predictor: ThreatPredictor) -> None:
        """A collector that exports only a few columns must still be scorable."""
        result = predictor.predict_one({"destination_port": 443, "protocol": 6}, explain=False)
        assert result.prediction in predictor.class_names

    def test_extra_columns_are_ignored(self, predictor: ThreatPredictor) -> None:
        flow = example_flow("benign")
        flow["some_unknown_vendor_column"] = 12345
        result = predictor.predict_one(flow, explain=False)
        assert result.prediction in predictor.class_names

    def test_predict_csv_preserves_row_count_and_columns(self, predictor: ThreatPredictor) -> None:
        frame = pd.DataFrame([example_flow("benign")] * 5)
        scored = predictor.predict_csv(frame)
        assert len(scored) == 5
        for column in ("prediction", "confidence", "risk_score", "risk_level", "is_malicious"):
            assert column in scored.columns

    def test_summarize_counts_are_coherent(self, predictor: ThreatPredictor) -> None:
        frame = pd.DataFrame([example_flow(k) for k in ("benign", "scan", "flood")] * 4)
        summary = predictor.summarize(predictor.predict_csv(frame))
        assert summary["total_records"] == 12
        assert summary["benign_records"] + summary["malicious_records"] == 12
        assert 0.0 <= summary["attack_rate"] <= 1.0

    def test_model_info_reports_contract(self, predictor: ThreatPredictor) -> None:
        info = predictor.model_info()
        assert info["supported_classes"] == predictor.class_names
        assert info["feature_count"] == len(predictor.feature_columns)


class TestRiskScoring:
    @pytest.mark.parametrize(
        ("score", "expected"),
        [(0, "LOW"), (30, "LOW"), (31, "MEDIUM"), (60, "MEDIUM"),
         (61, "HIGH"), (80, "HIGH"), (81, "CRITICAL"), (100, "CRITICAL")],
    )
    def test_band_boundaries(self, score: int, expected: str) -> None:
        assert risk_level_for(score) == expected

    def test_score_is_always_bounded(self) -> None:
        for predicted in ("BENIGN", "DDoS", "Infiltration", "PortScan"):
            for confidence in (0.0, 0.5, 1.0):
                for anomaly in (0.0, 0.5, 1.0):
                    assessment = compute_risk(predicted, confidence, anomaly, anomaly > 0.5)
                    assert 0 <= assessment.risk_score <= 100

    def test_benign_scores_lower_than_attack_at_equal_confidence(self) -> None:
        benign = compute_risk(BENIGN_LABEL, 0.95, 0.05, False)
        attack = compute_risk("DDoS", 0.95, 0.05, False)
        assert benign.risk_score < attack.risk_score

    def test_severity_ordering_is_respected(self) -> None:
        """A higher-severity family must not score below a lower one, all else equal."""
        scan = compute_risk("PortScan", 0.9, 0.2, False).risk_score
        infiltration = compute_risk("Infiltration", 0.9, 0.2, False).risk_score
        assert CLASS_SEVERITY["Infiltration"] > CLASS_SEVERITY["PortScan"]
        assert infiltration > scan

    def test_anomaly_flag_raises_floor_on_benign_prediction(self) -> None:
        """The unknown-attack path: an anomalous 'benign' flow cannot be dismissed."""
        normal = compute_risk(BENIGN_LABEL, 0.99, 0.02, is_anomaly=False)
        flagged = compute_risk(BENIGN_LABEL, 0.99, 0.9, is_anomaly=True)
        assert flagged.risk_score > normal.risk_score
        assert flagged.risk_score >= 45

    def test_components_are_reported(self) -> None:
        assessment = compute_risk("DDoS", 0.9, 0.8, True)
        assert set(assessment.components) == {
            "class_severity", "model_confidence", "anomaly", "traffic_characteristics"
        }


class TestAnomalyDetection:
    def test_scores_are_normalised(self, predictor: ThreatPredictor) -> None:
        if predictor.anomaly is None:
            pytest.skip("anomaly detector not available")
        frame = pd.DataFrame([example_flow(k) for k in ("benign", "scan", "flood")])
        results = predictor.predict_frame(frame)
        for result in results:
            assert 0.0 <= result.anomaly_score <= 1.0

    def test_flag_and_score_are_consistent(self, predictor: ThreatPredictor) -> None:
        if predictor.anomaly is None:
            pytest.skip("anomaly detector not available")
        results = predictor.predict_frame(pd.DataFrame([example_flow("benign")] * 3))
        for result in results:
            assert isinstance(result.is_anomaly, bool)

    def test_predictor_works_without_anomaly_detector(self, trained_bundle: ModelBundle) -> None:
        """Graceful degradation: no detector must not break supervised serving."""
        bare = ThreatPredictor(
            model_bundle=trained_bundle,
            use_anomaly_detector=False,
            enable_explanations=False,
        )
        result = bare.predict_one(example_flow("flood"), explain=False)
        assert result.anomaly_score == 0.0
        assert result.is_anomaly is False
