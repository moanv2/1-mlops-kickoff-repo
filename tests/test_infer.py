"""Tests for src.infer module."""

import numpy as np
import pandas as pd
import pytest

from src.infer import run_inference


class _FakeModel:
    """Minimal stub that implements predict."""

    def predict(self, X):
        return np.zeros(len(X), dtype=int)


class _FakeModelWithProba(_FakeModel):
    """Stub that also implements predict_proba."""

    def predict_proba(self, X):
        n = len(X)
        return np.column_stack([np.ones(n) * 0.8, np.ones(n) * 0.2])


class TestRunInference:
    """Tests for run_inference."""

    def test_returns_dataframe_with_prediction_column(self):
        model = _FakeModel()
        X = pd.DataFrame({"f1": [1, 2, 3]})
        result = run_inference(model, X)
        assert isinstance(result, pd.DataFrame)
        assert "prediction" in result.columns

    def test_preserves_index(self):
        model = _FakeModel()
        X = pd.DataFrame({"f1": [10, 20]}, index=[5, 9])
        result = run_inference(model, X)
        assert list(result.index) == [5, 9]

    def test_raises_on_non_dataframe_input(self):
        model = _FakeModel()
        with pytest.raises(TypeError, match="must be a pandas DataFrame"):
            run_inference(model, [[1, 2], [3, 4]])

    def test_raises_on_model_without_predict(self):
        with pytest.raises(TypeError, match="does not implement .predict"):
            run_inference(object(), pd.DataFrame({"f1": [1]}))

    def test_works_with_predict_proba(self):
        model = _FakeModelWithProba()
        X = pd.DataFrame({"f1": [1, 2]})
        result = run_inference(model, X)
        assert result.shape[0] == 2
