"""Tests for src.main module.

Tests the pipeline orchestration by verifying that main.py correctly
wires together all sub-modules (load, validate, clean, features, train,
evaluate, infer) and that its constants are consistent.
"""

import os
import subprocess
import sys

import pandas as pd
import pytest

from src.main import (
    REQUIRED_COLUMNS,
    TARGET_COL,
    PROBLEM_TYPE,
    MODEL_PATH,
)
from src.clean_data import clean_data
from src.validate import validate_dataframe
from src.feature_engineering import build_features, FeatureConfig
from src.train import train_model
from src.evaluate import evaluate_model
from src.infer import run_inference
from sklearn.preprocessing import FunctionTransformer


def _make_raw_df(n=80):
    """Build a synthetic DataFrame mimicking the raw telecom churn dataset."""
    import numpy as np
    rng = np.random.default_rng(42)
    return pd.DataFrame({
        "AccountWeeks": rng.integers(1, 200, n),
        "DataUsage": rng.uniform(0, 5, n).round(2),
        "CustServCalls": rng.integers(0, 10, n),
        "DayMins": rng.uniform(50, 350, n).round(1),
        "DayCalls": rng.integers(50, 150, n),
        "MonthlyCharge": rng.uniform(20, 100, n).round(2),
        "OverageFee": rng.uniform(0, 20, n).round(2),
        "RoamMins": rng.uniform(0, 20, n).round(1),
        "Churn": rng.choice([0, 1], n),
        "ContractRenewal": rng.choice([0, 1], n),
        "DataPlan": rng.choice([0, 1], n),
    })


# ---------------------------------------------------------------------------
# Unit tests (mock data, no disk I/O)
# ---------------------------------------------------------------------------


class TestMainConstants:
    """Verify main.py exports are internally consistent."""

    def test_required_columns_includes_target(self):
        assert "Churn" in REQUIRED_COLUMNS

    def test_target_col_is_lowercase(self):
        assert TARGET_COL == TARGET_COL.lower()

    def test_problem_type_is_valid(self):
        assert PROBLEM_TYPE in ("classification", "regression")


class TestPipelineIntegration:
    """End-to-end integration tests using mock data (no disk I/O)."""

    def test_validate_passes_on_clean_mock_data(self):
        df = _make_raw_df()
        assert validate_dataframe(df, REQUIRED_COLUMNS) is True

    def test_clean_data_standardizes_columns(self):
        df = _make_raw_df()
        cleaned = clean_data(df)
        for col in cleaned.columns:
            assert col == col.lower(), f"Column '{col}' not lowercased"

    def test_feature_engineering_preserves_target(self):
        df = _make_raw_df()
        cleaned = clean_data(df)
        cfg = FeatureConfig(
            target_col=TARGET_COL,
            numeric_cols=(
                "accountweeks", "datausage", "custservcalls",
                "daymins", "daycalls", "monthlycharge",
                "overagefee", "roammins",
            ),
            categorical_cols=("contractrenewal", "dataplan"),
        )
        featured = build_features(cleaned, cfg)
        assert TARGET_COL in featured.columns

    def test_full_pipeline_runs_end_to_end(self, tmp_path):
        """Smoke test: the entire pipeline runs without error on mock data."""
        df = _make_raw_df(n=100)

        # Validate
        validate_dataframe(df, REQUIRED_COLUMNS)

        # Clean
        df = clean_data(df)

        # Feature engineering
        cfg = FeatureConfig(
            target_col=TARGET_COL,
            numeric_cols=(
                "accountweeks", "datausage", "custservcalls",
                "daymins", "daycalls", "monthlycharge",
                "overagefee", "roammins",
            ),
            categorical_cols=("contractrenewal", "dataplan"),
        )
        df = build_features(df, cfg)

        # Split
        y = df[TARGET_COL]
        X = df.drop(columns=[TARGET_COL])

        # Train
        model_path = str(tmp_path / "model.pkl")
        preprocessor = FunctionTransformer()
        pipeline, X_val, y_val, X_test, y_test = train_model(
            X, y, preprocessor, PROBLEM_TYPE, model_path
        )

        # Evaluate
        metric = evaluate_model(pipeline, X_test, y_test, PROBLEM_TYPE)
        assert isinstance(metric, float)
        assert 0.0 <= metric <= 1.0

        # Infer
        preds = run_inference(pipeline, X_test)
        assert isinstance(preds, pd.DataFrame)
        assert "prediction" in preds.columns
        assert len(preds) == len(X_test)

    def test_pipeline_rejects_empty_dataframe(self):
        df = pd.DataFrame({col: [] for col in REQUIRED_COLUMNS})
        with pytest.raises(ValueError, match="empty"):
            validate_dataframe(df, REQUIRED_COLUMNS)

    def test_pipeline_rejects_missing_columns(self):
        df = _make_raw_df()
        df = df.drop(columns=["Churn"])
        with pytest.raises(ValueError, match="Missing required columns"):
            validate_dataframe(df, REQUIRED_COLUMNS)


# ---------------------------------------------------------------------------
# Subprocess tests (run main.py as a real process against real data)
# ---------------------------------------------------------------------------


class TestMainPipeline:
    """Tests for the main.py pipeline orchestrator via subprocess."""

    def test_pipeline_runs_end_to_end(self):
        """The full pipeline should complete without errors."""
        result = subprocess.run(
            [sys.executable, "-m", "src.main"],
            capture_output=True,
            text=True,
            timeout=120,
        )
        assert result.returncode == 0, (
            f"Pipeline failed with stderr:\n{result.stderr}"
        )

    def test_pipeline_outputs_metric(self):
        """The pipeline should print the final metric."""
        result = subprocess.run(
            [sys.executable, "-m", "src.main"],
            capture_output=True,
            text=True,
            timeout=120,
        )
        assert "Pipeline complete" in result.stdout
        assert "metric" in result.stdout

    def test_pipeline_outputs_predictions(self):
        """The pipeline should print prediction results."""
        result = subprocess.run(
            [sys.executable, "-m", "src.main"],
            capture_output=True,
            text=True,
            timeout=120,
        )
        assert "prediction" in result.stdout

    def test_pipeline_produces_model_artifact(self):
        """The pipeline should save a model file."""
        result = subprocess.run(
            [sys.executable, "-m", "src.main"],
            capture_output=True,
            text=True,
            timeout=120,
        )
        assert result.returncode == 0
        assert os.path.exists("models/model.pkl")

    def test_pipeline_produces_confusion_matrix(self):
        """The pipeline should save a confusion matrix plot."""
        result = subprocess.run(
            [sys.executable, "-m", "src.main"],
            capture_output=True,
            text=True,
            timeout=120,
        )
        assert result.returncode == 0
        assert os.path.exists("reports/figures/confusion_matrix.png")
