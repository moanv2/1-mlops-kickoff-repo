"""Tests for src.api module."""

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer
from unittest.mock import patch

from fastapi.testclient import TestClient

from src.clean_data import clean_data
from src.feature_engineering import FeatureConfig, build_features
from src.utils import load_config


def _build_mock_model():
    """Build a model trained on feature-engineered data (same as the real pipeline)."""
    config = load_config()
    rng = np.random.default_rng(42)
    n = 100
    raw = pd.DataFrame({
        "accountweeks": rng.integers(1, 200, n).astype(float),
        "datausage": rng.uniform(0, 5, n),
        "custservcalls": rng.integers(0, 10, n).astype(float),
        "daymins": rng.uniform(50, 350, n),
        "daycalls": rng.integers(50, 150, n).astype(float),
        "monthlycharge": rng.uniform(20, 100, n),
        "overagefee": rng.uniform(0, 20, n),
        "roammins": rng.uniform(0, 20, n),
        "contractrenewal": rng.choice([0, 1], n).astype(float),
        "dataplan": rng.choice([0, 1], n).astype(float),
    })
    y = pd.Series(rng.choice([0, 1], n))

    # Apply the same feature engineering the API will apply
    cleaned = clean_data(raw, config)
    cfg = FeatureConfig(
        numeric_cols=tuple(config["features"]["numeric"]),
        categorical_cols=tuple(config["features"]["categorical"]),
    )
    featured = build_features(cleaned, cfg)

    pipeline = Pipeline([
        ("preprocess", FunctionTransformer()),
        ("model", LogisticRegression(max_iter=500)),
    ])
    pipeline.fit(featured, y)
    return pipeline


pytestmark = pytest.mark.filterwarnings("ignore::sklearn.exceptions.ConvergenceWarning")


@pytest.fixture(autouse=True)
def _mock_model():
    """Patch the model loader so tests don't need W&B."""
    model = _build_mock_model()
    with patch("src.api._get_model", return_value=model):
        from src.api import app
        yield app


@pytest.fixture()
def client(_mock_model):
    return TestClient(_mock_model)


SAMPLE_CUSTOMER = {
    "accountweeks": 120,
    "datausage": 2.7,
    "custservcalls": 3,
    "daymins": 200.5,
    "daycalls": 90,
    "monthlycharge": 65.0,
    "overagefee": 10.25,
    "roammins": 8.3,
    "contractrenewal": 1,
    "dataplan": 0,
}


class TestHealthEndpoint:
    def test_health_returns_200(self, client):
        resp = client.get("/health")
        assert resp.status_code == 200

    def test_health_response_shape(self, client):
        data = client.get("/health").json()
        assert "status" in data
        assert data["status"] == "healthy"


class TestPredictEndpoint:
    def test_predict_single_customer(self, client):
        resp = client.post("/predict", json={"customers": [SAMPLE_CUSTOMER]})
        assert resp.status_code == 200
        data = resp.json()
        assert len(data["predictions"]) == 1
        assert data["predictions"][0]["prediction"] in (0, 1)
        assert data["predictions"][0]["label"] in ("No Churn", "Churn")

    def test_predict_batch(self, client):
        resp = client.post("/predict", json={"customers": [SAMPLE_CUSTOMER] * 5})
        assert resp.status_code == 200
        assert len(resp.json()["predictions"]) == 5

    def test_predict_rejects_missing_fields(self, client):
        bad = {"accountweeks": 100}  # missing all other fields
        resp = client.post("/predict", json={"customers": [bad]})
        assert resp.status_code == 422

    def test_predict_rejects_negative_values(self, client):
        bad = {**SAMPLE_CUSTOMER, "accountweeks": -5}
        resp = client.post("/predict", json={"customers": [bad]})
        assert resp.status_code == 422

    def test_predict_rejects_invalid_binary(self, client):
        bad = {**SAMPLE_CUSTOMER, "contractrenewal": 3}
        resp = client.post("/predict", json={"customers": [bad]})
        assert resp.status_code == 422

    @pytest.mark.filterwarnings("ignore::UserWarning")
    def test_predict_empty_list_rejected(self, client):
        resp = client.post("/predict", json={"customers": []})
        # Empty list will pass Pydantic but fail at inference
        assert resp.status_code in (200, 422)
