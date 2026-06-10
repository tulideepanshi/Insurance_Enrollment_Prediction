"""
Tests for src.api.app module.

Covers: health endpoint, single prediction, batch prediction,
input validation, error handling, drift warnings.
"""

import json

import pytest
from fastapi.testclient import TestClient
from sklearn.pipeline import Pipeline

from src.api.app import app
from src.features.feature_engineering import build_feature_pipeline
from src.models.trainer import get_candidate_models


@pytest.fixture(autouse=True)
def load_test_model(sample_features_df, sample_target, tmp_path):
    """Train a lightweight model and inject it into the API module.

    Also creates a known_categories.json for drift detection testing.
    """
    import src.api.app as api_module

    model = get_candidate_models()["logistic_regression"]
    pipeline = Pipeline([
        ("features", build_feature_pipeline("logistic_regression")),
        ("classifier", model),
    ])
    pipeline.fit(sample_features_df, sample_target)
    api_module._model_pipeline = pipeline

    # Set up known categories for drift detection
    safe_encoder = pipeline.named_steps["features"].named_steps["safe_encoder"]
    api_module._known_categories = safe_encoder.get_known_categories()

    yield
    api_module._model_pipeline = None
    api_module._known_categories = {}


@pytest.fixture
def client():
    """FastAPI test client."""
    return TestClient(app, raise_server_exceptions=False)


@pytest.fixture
def valid_employee():
    """A valid employee input payload."""
    return {
        "age": 35,
        "gender": "Female",
        "marital_status": "Married",
        "salary": 72000.0,
        "employment_type": "Full-time",
        "region": "Northeast",
        "has_dependents": "Yes",
        "tenure_years": 5.2,
    }


class TestHealthEndpoint:
    """Tests for GET /health."""

    def test_health_returns_200(self, client):
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert data["model_loaded"] is True


class TestPredictEndpoint:
    """Tests for POST /predict."""

    def test_valid_prediction(self, client, valid_employee):
        """Should return enrolled prediction, probability, and empty warnings."""
        response = client.post("/predict", json=valid_employee)
        assert response.status_code == 200
        data = response.json()
        assert "enrolled" in data
        assert "probability" in data
        assert "warnings" in data
        assert data["enrolled"] in [0, 1]
        assert 0 <= data["probability"] <= 1
        assert data["warnings"] == []

    def test_invalid_gender_returns_422(self, client, valid_employee):
        """Invalid categorical value should trigger validation error."""
        valid_employee["gender"] = "Invalid"
        response = client.post("/predict", json=valid_employee)
        assert response.status_code == 422

    def test_negative_salary_returns_422(self, client, valid_employee):
        """Negative salary should trigger validation error."""
        valid_employee["salary"] = -1000
        response = client.post("/predict", json=valid_employee)
        assert response.status_code == 422

    def test_age_below_minimum_returns_422(self, client, valid_employee):
        """Age below 18 should trigger validation error."""
        valid_employee["age"] = 10
        response = client.post("/predict", json=valid_employee)
        assert response.status_code == 422

    def test_missing_field_returns_422(self, client, valid_employee):
        """Missing required field should trigger validation error."""
        del valid_employee["salary"]
        response = client.post("/predict", json=valid_employee)
        assert response.status_code == 422

    def test_invalid_employment_type_returns_422(self, client, valid_employee):
        """Invalid employment type should be rejected."""
        valid_employee["employment_type"] = "Freelance"
        response = client.post("/predict", json=valid_employee)
        assert response.status_code == 422

    def test_invalid_region_returns_422(self, client, valid_employee):
        """Invalid region should be rejected."""
        valid_employee["region"] = "Europe"
        response = client.post("/predict", json=valid_employee)
        assert response.status_code == 422


class TestBatchPredictEndpoint:
    """Tests for POST /predict/batch."""

    def test_batch_prediction(self, client, valid_employee):
        """Should return predictions for each input record."""
        payload = [valid_employee, valid_employee]
        response = client.post("/predict/batch", json=payload)
        assert response.status_code == 200
        data = response.json()
        assert data["count"] == 2
        assert len(data["predictions"]) == 2
        assert "warnings" in data

    def test_empty_batch_returns_400(self, client):
        """Empty input should return 400."""
        response = client.post("/predict/batch", json=[])
        assert response.status_code == 400


class TestModelInfoEndpoint:
    """Tests for GET /model/info."""

    def test_model_info(self, client):
        """Should return model type and pipeline steps."""
        response = client.get("/model/info")
        assert response.status_code == 200
        data = response.json()
        assert "model_type" in data
        assert "pipeline_steps" in data
        assert isinstance(data["pipeline_steps"], list)


class TestDriftDetection:
    """Tests for drift warning behavior."""

    def test_no_warnings_for_known_values(self, client, valid_employee):
        """Known values should produce no warnings."""
        response = client.post("/predict", json=valid_employee)
        assert response.status_code == 200
        assert response.json()["warnings"] == []

    def test_batch_drift_warnings_aggregated(self, client, valid_employee):
        """Batch endpoint should aggregate drift warnings."""
        # All valid — no warnings
        payload = [valid_employee, valid_employee]
        response = client.post("/predict/batch", json=payload)
        assert response.status_code == 200
        assert response.json()["warnings"] == []


class TestModelNotLoaded:
    """Tests for behavior when model is not loaded."""

    def test_predict_without_model_returns_503(self, client, valid_employee):
        """Should return 503 when model pipeline is None."""
        import src.api.app as api_module
        api_module._model_pipeline = None
        response = client.post("/predict", json=valid_employee)
        assert response.status_code == 503
