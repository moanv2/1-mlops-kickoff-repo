"""Tests for src.evaluate module."""

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.pipeline import Pipeline

from src.evaluate import evaluate_model


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def regression_fixtures():
    """Fitted LinearRegression pipeline + aligned X_test / y_test."""
    rng = np.random.default_rng(0)
    X = pd.DataFrame({"a": rng.random(40), "b": rng.random(40)})
    y = pd.Series(X["a"] * 2 + rng.normal(0, 0.05, 40))
    pipeline = Pipeline([("model", LinearRegression())])
    pipeline.fit(X, y)
    return pipeline, X, y


@pytest.fixture()
def classification_fixtures():
    """Fitted LogisticRegression pipeline + aligned X_test / y_test."""
    rng = np.random.default_rng(1)
    X = pd.DataFrame({"a": rng.random(40), "b": rng.random(40)})
    y = pd.Series((X["a"] > 0.5).astype(int))
    pipeline = Pipeline([("model", LogisticRegression())])
    pipeline.fit(X, y)
    return pipeline, X, y


# ---------------------------------------------------------------------------
# Happy-path tests
# ---------------------------------------------------------------------------


class TestEvaluateModel:
    def test_regression_returns_float(self, regression_fixtures):
        model, X, y = regression_fixtures
        result = evaluate_model(model, X, y, "regression")
        assert isinstance(result, float)

    def test_regression_rmse_non_negative(self, regression_fixtures):
        model, X, y = regression_fixtures
        result = evaluate_model(model, X, y, "regression")
        assert result >= 0.0

    def test_classification_returns_float(self, classification_fixtures):
        model, X, y = classification_fixtures
        result = evaluate_model(model, X, y, "classification")
        assert isinstance(result, float)

    def test_classification_f1_in_unit_interval(self, classification_fixtures):
        model, X, y = classification_fixtures
        result = evaluate_model(model, X, y, "classification")
        assert 0.0 <= result <= 1.0

    def test_regression_saves_residual_plot(self, tmp_path, monkeypatch, regression_fixtures):
        """evaluate_model saves residual_plot.png to REPORTS_DIR."""
        import src.evaluate as ev
        monkeypatch.setattr(ev, "REPORTS_DIR", tmp_path)
        model, X, y = regression_fixtures
        evaluate_model(model, X, y, "regression")
        assert (tmp_path / "residual_plot.png").exists()

    def test_classification_saves_confusion_matrix(self, tmp_path, monkeypatch, classification_fixtures):
        """evaluate_model saves confusion_matrix.png to REPORTS_DIR."""
        import src.evaluate as ev
        monkeypatch.setattr(ev, "REPORTS_DIR", tmp_path)
        model, X, y = classification_fixtures
        evaluate_model(model, X, y, "classification")
        assert (tmp_path / "confusion_matrix.png").exists()


# ---------------------------------------------------------------------------
# Guardrail tests
# ---------------------------------------------------------------------------


class TestEvaluateModelGuardrails:
    def test_raises_if_model_has_no_predict(self):
        X = pd.DataFrame({"a": [1, 2, 3]})
        y = pd.Series([1, 2, 3])
        with pytest.raises(ValueError, match="predict"):
            evaluate_model(object(), X, y, "regression")

    def test_raises_on_empty_x_test(self, regression_fixtures):
        model, _, _ = regression_fixtures
        X_empty = pd.DataFrame({"a": [], "b": []})
        y_empty = pd.Series([], dtype=float)
        with pytest.raises(ValueError, match="empty"):
            evaluate_model(model, X_empty, y_empty, "regression")

    def test_raises_on_length_mismatch(self, regression_fixtures):
        model, X, y = regression_fixtures
        y_short = y.iloc[:-5]
        with pytest.raises(ValueError, match="same length"):
            evaluate_model(model, X, y_short, "regression")

    def test_raises_on_unknown_problem_type(self, regression_fixtures):
        model, X, y = regression_fixtures
        with pytest.raises(ValueError, match="Unknown problem_type"):
            evaluate_model(model, X, y, "clustering")
