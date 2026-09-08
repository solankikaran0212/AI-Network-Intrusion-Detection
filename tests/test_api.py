"""End-to-end tests for the FastAPI service.

The app is exercised through Starlette's ``TestClient``, which runs the real
routing, dependency-injection, validation and serialisation stack. The model
dependency is overridden with the session-trained predictor so these tests do
not depend on artifacts existing in the repository.
"""

from __future__ import annotations

import io

import pandas as pd
import pytest
from fastapi.testclient import TestClient

from api.main import app, require_predictor
from src.predict import ThreatPredictor, example_flow


@pytest.fixture(scope="module")
def client(predictor: ThreatPredictor):
    """TestClient with the model dependency wired to the test predictor."""
    import api.main as api_main

    app.dependency_overrides[require_predictor] = lambda: predictor
    api_main._PREDICTOR = predictor  # health/lifespan read the module global
    with TestClient(app) as test_client:
        yield test_client
    app.dependency_overrides.clear()


class TestHealth:
    def test_health_returns_200(self, client: TestClient) -> None:
        response = client.get("/health")
        assert response.status_code == 200

    def test_health_reports_loaded_model(self, client: TestClient) -> None:
        body = client.get("/health").json()
        assert body["status"] == "healthy"
        assert body["model_loaded"] is True
        assert "uptime_seconds" in body

    def test_root_points_at_docs(self, client: TestClient) -> None:
        assert client.get("/").json()["docs"] == "/docs"

    def test_openapi_schema_is_valid(self, client: TestClient) -> None:
        schema = client.get("/openapi.json").json()
        for path in ("/health", "/predict", "/predict/batch", "/model-info"):
            assert path in schema["paths"]


class TestModelInfo:
    def test_returns_model_metadata(self, client: TestClient) -> None:
        body = client.get("/model-info").json()
        assert body["model_name"] in {"LogisticRegression", "RandomForest", "XGBoost"}
        assert body["feature_count"] > 0
        assert isinstance(body["supported_classes"], list)
        assert body["supported_classes"]


class TestPredict:
    def test_valid_flow_returns_full_payload(self, client: TestClient) -> None:
        response = client.post("/predict", json=example_flow("flood"))
        assert response.status_code == 200
        body = response.json()
        for field in ("prediction", "confidence", "risk_score", "risk_level",
                      "is_anomaly", "anomaly_score", "is_malicious", "class_probabilities"):
            assert field in body

    def test_response_values_are_in_range(self, client: TestClient) -> None:
        body = client.post("/predict", json=example_flow("benign")).json()
        assert 0.0 <= body["confidence"] <= 1.0
        assert 0 <= body["risk_score"] <= 100
        assert body["risk_level"] in {"LOW", "MEDIUM", "HIGH", "CRITICAL"}

    def test_minimal_payload_is_accepted(self, client: TestClient) -> None:
        """Only destination_port is required; the rest is imputed."""
        response = client.post("/predict", json={"destination_port": 443})
        assert response.status_code == 200

    def test_unknown_extra_fields_are_ignored(self, client: TestClient) -> None:
        flow = example_flow("benign") | {"vendor_specific_column": 99}
        assert client.post("/predict", json=flow).status_code == 200

    def test_explain_flag_controls_attributions(self, client: TestClient) -> None:
        without = client.post("/predict?explain=false", json=example_flow("scan")).json()
        assert without["top_features"] == []

    @pytest.mark.parametrize(
        "payload",
        [
            {},                                              # missing required port
            {"destination_port": 70000},                     # port above 16-bit range
            {"destination_port": -1},                         # negative port
            {"destination_port": 80, "total_fwd_packets": -5},  # negative count
            {"destination_port": 80, "protocol": 999},        # protocol out of range
            {"destination_port": "not-a-number"},             # wrong type
        ],
    )
    def test_invalid_input_is_rejected_with_422(self, client: TestClient, payload: dict) -> None:
        assert client.post("/predict", json=payload).status_code == 422

    def test_non_finite_values_are_rejected(self, client: TestClient) -> None:
        """JSON has no NaN literal, but Python's parser accepts it."""
        response = client.post(
            "/predict",
            content='{"destination_port": 80, "flow_bytes_s": Infinity}',
            headers={"Content-Type": "application/json"},
        )
        assert response.status_code == 422

    def test_top_k_is_bounded(self, client: TestClient) -> None:
        assert client.post("/predict?top_k=0", json=example_flow("benign")).status_code == 422
        assert client.post("/predict?top_k=99", json=example_flow("benign")).status_code == 422


class TestBatchPredict:
    @staticmethod
    def _csv_upload(rows: int = 20) -> io.BytesIO:
        frame = pd.DataFrame([example_flow("benign"), example_flow("flood")] * (rows // 2))
        buffer = io.BytesIO()
        frame.to_csv(buffer, index=False)
        buffer.seek(0)
        return buffer

    def test_valid_csv_is_scored(self, client: TestClient) -> None:
        response = client.post(
            "/predict/batch", files={"file": ("flows.csv", self._csv_upload(), "text/csv")}
        )
        assert response.status_code == 200
        body = response.json()
        assert body["rows_scored"] == 20
        assert body["summary"]["total_records"] == 20

    def test_summary_counts_are_coherent(self, client: TestClient) -> None:
        body = client.post(
            "/predict/batch", files={"file": ("flows.csv", self._csv_upload(), "text/csv")}
        ).json()
        summary = body["summary"]
        assert summary["benign_records"] + summary["malicious_records"] == summary["total_records"]
        assert 0.0 <= summary["attack_rate"] <= 1.0

    def test_preview_rows_bounds_the_response(self, client: TestClient) -> None:
        body = client.post(
            "/predict/batch?preview_rows=5",
            files={"file": ("flows.csv", self._csv_upload(), "text/csv")},
        ).json()
        assert len(body["predictions"]) == 5
        assert body["truncated"] is True

    def test_non_csv_extension_is_rejected(self, client: TestClient) -> None:
        response = client.post(
            "/predict/batch", files={"file": ("payload.txt", io.BytesIO(b"a,b\n1,2"), "text/plain")}
        )
        assert response.status_code == 400

    def test_empty_file_is_rejected(self, client: TestClient) -> None:
        response = client.post(
            "/predict/batch", files={"file": ("empty.csv", io.BytesIO(b""), "text/csv")}
        )
        assert response.status_code == 400

    def test_unparseable_content_is_rejected(self, client: TestClient) -> None:
        blob = io.BytesIO(b"\x80\x81\x82 not valid utf-8")
        response = client.post(
            "/predict/batch", files={"file": ("bad.csv", blob, "text/csv")}
        )
        assert response.status_code == 400

    def test_header_only_csv_is_rejected(self, client: TestClient) -> None:
        response = client.post(
            "/predict/batch",
            files={"file": ("head.csv", io.BytesIO(b"destination_port,protocol\n"), "text/csv")},
        )
        assert response.status_code == 400

    def test_missing_file_field_is_422(self, client: TestClient) -> None:
        assert client.post("/predict/batch").status_code == 422


class TestDegradedMode:
    def test_prediction_returns_503_without_a_model(self) -> None:
        """A deployment with no trained model must fail closed, not crash."""
        import api.main as api_main

        overrides = dict(app.dependency_overrides)
        app.dependency_overrides.clear()
        original = api_main._PREDICTOR
        api_main._PREDICTOR = None
        try:
            with TestClient(app) as bare:
                api_main._PREDICTOR = None  # lifespan may have reloaded it
                assert bare.post("/predict", json=example_flow("benign")).status_code == 503
                assert bare.get("/model-info").status_code == 503
                assert bare.get("/health").status_code == 200
        finally:
            api_main._PREDICTOR = original
            app.dependency_overrides.update(overrides)
