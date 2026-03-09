"""Tests for src.train module."""

import os

import numpy as np
import pandas as pd
import pytest
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.train import train_model


def _make_preprocessor():
    return ColumnTransformer([
        ("num", StandardScaler(), ["age", "charges"]),
        ("cat", OneHotEncoder(), ["plan"]),
    ])


def _make_data(n=100):
    np.random.seed(42)
    X = pd.DataFrame({
        "age": np.random.randint(18, 70, n).astype(float),
        "charges": np.random.uniform(20, 100, n),
        "plan": np.random.choice(["basic", "premium"], n),
    })
    y = pd.Series(np.random.randint(0, 2, n), name="churn")
    return X, y


class TestTrainModel:
    def test_returns_tuple_of_five(self, tmp_path):
        X, y = _make_data()
        result = train_model(
            X, y, _make_preprocessor(), "classification",
            model_path=str(tmp_path / "model.pkl"),
            test_size=0.2,
            val_size=0.2,
        )
        assert isinstance(result, tuple) and len(result) == 5

    def test_first_element_is_fitted_pipeline(self, tmp_path):
        X, y = _make_data()
        pipeline, _, _, _, _ = train_model(
            X, y, _make_preprocessor(), "classification",
            model_path=str(tmp_path / "model.pkl"),
            test_size=0.2,
            val_size=0.2,
        )
        assert isinstance(pipeline, Pipeline)
        assert [s[0] for s in pipeline.steps] == ["preprocess", "model"]

    def test_split_shapes_are_consistent(self, tmp_path):
        X, y = _make_data(n=100)
        _, X_val, y_val, X_test, y_test = train_model(
            X, y, _make_preprocessor(), "classification",
            model_path=str(tmp_path / "model.pkl"),
            test_size=0.2,
            val_size=0.2,
            random_state=42,
        )
        assert len(X_val) == len(y_val)
        assert len(X_test) == len(y_test)
        assert len(X_val) == 20  # 20% of 100
        assert len(X_test) == 20  # 20% of 100

    def test_artifact_is_saved_to_disk(self, tmp_path):
        X, y = _make_data()
        model_path = str(tmp_path / "model.pkl")
        train_model(
            X, y, _make_preprocessor(), "classification",
            model_path=model_path,
            test_size=0.2,
            val_size=0.2,
        )
        assert os.path.exists(model_path)

    def test_artifact_directory_is_created_if_missing(self, tmp_path):
        X, y = _make_data()
        model_path = str(tmp_path / "subdir" / "model.pkl")
        train_model(
            X, y, _make_preprocessor(), "classification",
            model_path=model_path,
            test_size=0.2,
            val_size=0.2,
        )
        assert os.path.exists(model_path)

    def test_fitted_pipeline_can_predict(self, tmp_path):
        X, y = _make_data()
        pipeline, _, _, X_test, _ = train_model(
            X, y, _make_preprocessor(), "classification",
            model_path=str(tmp_path / "model.pkl"),
            test_size=0.2,
            val_size=0.2,
        )
        preds = pipeline.predict(X_test)
        assert len(preds) == len(X_test)

    def test_regression_uses_ridge(self, tmp_path):
        X, y = _make_data()
        y = pd.Series(np.random.uniform(0, 1, len(y)))
        pipeline, _, _, _, _ = train_model(
            X, y, _make_preprocessor(), "regression",
            model_path=str(tmp_path / "model.pkl"),
            test_size=0.2,
            val_size=0.2,
        )
        assert pipeline.named_steps["model"].__class__.__name__ == "Ridge"

    def test_classification_uses_logistic_regression(self, tmp_path):
        X, y = _make_data()
        pipeline, _, _, _, _ = train_model(
            X, y, _make_preprocessor(), "classification",
            model_path=str(tmp_path / "model.pkl"),
            test_size=0.2,
            val_size=0.2,
        )
        assert pipeline.named_steps["model"].__class__.__name__ == "LogisticRegression"

    def test_raises_on_empty_X(self, tmp_path):
        X = pd.DataFrame({"age": [], "charges": [], "plan": []})
        y = pd.Series([], dtype=int)
        with pytest.raises(ValueError, match="X is empty"):
            train_model(
                X, y, _make_preprocessor(), "classification",
                model_path=str(tmp_path / "model.pkl"),
                test_size=0.2,
                val_size=0.2,
            )

    def test_raises_on_shape_mismatch(self, tmp_path):
        X, y = _make_data(n=50)
        y_short = y.iloc[:40]
        with pytest.raises(ValueError, match="Shape mismatch"):
            train_model(
                X, y_short, _make_preprocessor(), "classification",
                model_path=str(tmp_path / "model.pkl"),
                test_size=0.2,
                val_size=0.2,
            )

    def test_raises_on_unsupported_problem_type(self, tmp_path):
        X, y = _make_data()
        with pytest.raises(ValueError, match="Unsupported problem_type|Unsupported problem_type|Expected"):
            train_model(
                X, y, _make_preprocessor(), "clustering",
                model_path=str(tmp_path / "model.pkl"),
                test_size=0.2,
                val_size=0.2,
            )

    def test_random_state_produces_reproducible_test_split(self, tmp_path):
        X, y = _make_data()
        _, _, _, X_test_a, _ = train_model(
            X, y, _make_preprocessor(), "classification",
            model_path=str(tmp_path / "a.pkl"),
            random_state=7,
            test_size=0.2,
            val_size=0.2,
        )
        _, _, _, X_test_b, _ = train_model(
            X, y, _make_preprocessor(), "classification",
            model_path=str(tmp_path / "b.pkl"),
            random_state=7,
            test_size=0.2,
            val_size=0.2,
        )
        pd.testing.assert_frame_equal(
            X_test_a.reset_index(drop=True),
            X_test_b.reset_index(drop=True),
        )