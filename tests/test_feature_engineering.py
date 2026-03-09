"""Tests for src.feature_engineering module."""

import pandas as pd

from src.feature_engineering import (
    FeatureConfig,
    build_features,
    _parse_dates,
    _encode_categoricals,
    _clean_numerics,
)


# ---------------------------------------------------------------------------
# FeatureConfig
# ---------------------------------------------------------------------------

class TestFeatureConfig:
    """Tests for the FeatureConfig dataclass."""

    def test_defaults_are_empty(self):
        cfg = FeatureConfig()
        assert cfg.target_col is None
        assert cfg.drop_cols == ()
        assert cfg.numeric_cols == ()
        assert cfg.categorical_cols == ()
        assert cfg.date_cols == ()

    def test_custom_values(self):
        cfg = FeatureConfig(
            target_col="churn",
            numeric_cols=("age", "income"),
            categorical_cols=("plan",),
        )
        assert cfg.target_col == "churn"
        assert "age" in cfg.numeric_cols


# ---------------------------------------------------------------------------
# _parse_dates
# ---------------------------------------------------------------------------

class TestParseDates:
    """Tests for the _parse_dates helper."""

    def test_creates_year_month_day_dow_columns(self):
        df = pd.DataFrame({"date": ["2024-03-15", "2024-06-20"]})
        result = _parse_dates(df, ["date"])
        assert "date_year" in result.columns
        assert "date_month" in result.columns
        assert "date_day" in result.columns
        assert "date_dow" in result.columns

    def test_does_not_mutate_original(self):
        df = pd.DataFrame({"date": ["2024-01-01"]})
        _parse_dates(df, ["date"])
        assert "date_year" not in df.columns

    def test_skips_missing_columns(self):
        df = pd.DataFrame({"a": [1]})
        result = _parse_dates(df, ["nonexistent"])
        assert list(result.columns) == ["a"]

    def test_handles_invalid_dates_as_nat(self):
        df = pd.DataFrame({"date": ["not-a-date", "2024-01-01"]})
        result = _parse_dates(df, ["date"])
        assert pd.isna(result["date_year"].iloc[0])


# ---------------------------------------------------------------------------
# _encode_categoricals
# ---------------------------------------------------------------------------

class TestEncodeCategoricals:
    """Tests for the _encode_categoricals helper."""

    def test_one_hot_encodes_columns(self):
        df = pd.DataFrame({"plan": ["basic", "premium", "basic"]})
        result = _encode_categoricals(df, ["plan"])
        assert "plan" not in result.columns
        assert any("plan_" in c for c in result.columns)

    def test_does_not_mutate_original(self):
        df = pd.DataFrame({"plan": ["basic", "premium"]})
        _encode_categoricals(df, ["plan"])
        assert "plan" in df.columns

    def test_skips_missing_columns(self):
        df = pd.DataFrame({"a": [1, 2]})
        result = _encode_categoricals(df, ["nonexistent"])
        pd.testing.assert_frame_equal(result, df)

    def test_includes_dummy_na_column(self):
        df = pd.DataFrame({"plan": ["basic", None]})
        result = _encode_categoricals(df, ["plan"])
        na_cols = [c for c in result.columns if "nan" in c.lower()]
        assert len(na_cols) > 0


# ---------------------------------------------------------------------------
# _clean_numerics
# ---------------------------------------------------------------------------

class TestCleanNumerics:
    """Tests for the _clean_numerics helper."""

    def test_fills_na_with_median(self):
        df = pd.DataFrame({"age": [10.0, None, 30.0]})
        result = _clean_numerics(df, ["age"])
        assert result["age"].isna().sum() == 0
        assert result["age"].iloc[1] == 20.0  # median of 10 and 30

    def test_creates_missing_indicator(self):
        df = pd.DataFrame({"age": [10.0, None, 30.0]})
        result = _clean_numerics(df, ["age"])
        assert "age_is_missing" in result.columns
        assert result["age_is_missing"].iloc[1] == 1
        assert result["age_is_missing"].iloc[0] == 0

    def test_coerces_non_numeric_to_nan(self):
        df = pd.DataFrame({"age": ["25", "not_a_number", "30"]})
        result = _clean_numerics(df, ["age"])
        assert result["age_is_missing"].iloc[1] == 1

    def test_does_not_mutate_original(self):
        df = pd.DataFrame({"age": [1.0, None]})
        _clean_numerics(df, ["age"])
        assert df["age"].isna().sum() == 1

    def test_skips_missing_columns(self):
        df = pd.DataFrame({"a": [1, 2]})
        result = _clean_numerics(df, ["nonexistent"])
        assert list(result.columns) == ["a"]


# ---------------------------------------------------------------------------
# build_features (integration)
# ---------------------------------------------------------------------------

class TestBuildFeatures:
    """Tests for the build_features function."""

    def test_returns_dataframe(self):
        df = pd.DataFrame({"age": [25, 30], "churn": [0, 1]})
        cfg = FeatureConfig(target_col="churn", numeric_cols=("age",))
        result = build_features(df, cfg)
        assert isinstance(result, pd.DataFrame)

    def test_does_not_mutate_original(self):
        df = pd.DataFrame({"age": [25, 30], "churn": [0, 1]})
        cfg = FeatureConfig(target_col="churn", numeric_cols=("age",))
        build_features(df, cfg)
        assert list(df.columns) == ["age", "churn"]

    def test_drops_specified_columns(self):
        df = pd.DataFrame({"id": [1, 2], "age": [25, 30]})
        cfg = FeatureConfig(drop_cols=("id",), numeric_cols=("age",))
        result = build_features(df, cfg)
        assert "id" not in result.columns

    def test_preserves_target_column(self):
        df = pd.DataFrame({"age": [25, 30], "churn": [0, 1]})
        cfg = FeatureConfig(target_col="churn", numeric_cols=("age",))
        result = build_features(df, cfg)
        assert "churn" in result.columns

    def test_encodes_categoricals_and_cleans_numerics(self):
        df = pd.DataFrame({
            "age": [25, 30],
            "plan": ["basic", "premium"],
            "churn": [0, 1],
        })
        cfg = FeatureConfig(
            target_col="churn",
            numeric_cols=("age",),
            categorical_cols=("plan",),
        )
        result = build_features(df, cfg)
        assert "age_is_missing" in result.columns
        assert any("plan_" in c for c in result.columns)

    def test_empty_config_returns_copy(self):
        df = pd.DataFrame({"a": [1, 2], "b": [3, 4]})
        cfg = FeatureConfig()
        result = build_features(df, cfg)
        pd.testing.assert_frame_equal(result, df)

    def test_drop_nonexistent_column_is_safe(self):
        df = pd.DataFrame({"a": [1]})
        cfg = FeatureConfig(drop_cols=("nonexistent",))
        result = build_features(df, cfg)
        assert "a" in result.columns
