"""Tests for src.clean_data module."""

import pandas as pd

from src.clean_data import clean_data


class TestCleanData:
    """Tests for clean_data."""

    def test_returns_dataframe(self):
        df = pd.DataFrame({"A": [1, 2], "B": [3, 4]})
        result = clean_data(df)
        assert isinstance(result, pd.DataFrame)

    def test_does_not_mutate_original(self):
        df = pd.DataFrame({"A": [1, 2], "B": [3, 4]})
        original_cols = list(df.columns)
        clean_data(df)
        assert list(df.columns) == original_cols

    def test_lowercases_column_names(self):
        df = pd.DataFrame({"AccountWeeks": [10], "DataUsage": [5]})
        result = clean_data(df)
        assert list(result.columns) == ["accountweeks", "datausage"]

    def test_strips_whitespace_from_column_names(self):
        df = pd.DataFrame({" Name ": [1], " Age ": [2]})
        result = clean_data(df)
        assert "name" in result.columns
        assert "age" in result.columns

    def test_replaces_spaces_with_underscores_in_columns(self):
        df = pd.DataFrame({"First Name": [1], "Last Name": [2]})
        result = clean_data(df)
        assert "first_name" in result.columns
        assert "last_name" in result.columns

    def test_drops_duplicate_rows(self):
        df = pd.DataFrame({"a": [1, 1, 2], "b": [3, 3, 4]})
        result = clean_data(df)
        assert len(result) == 2

    def test_strips_whitespace_from_string_columns(self):
        df = pd.DataFrame({"name": ["  alice  ", " bob"]})
        result = clean_data(df)
        assert result["name"].iloc[0] == "alice"
        assert result["name"].iloc[1] == "bob"

    def test_replaces_sentinel_values_with_na(self):
        df = pd.DataFrame({"a": ["NA", "N/A", "?", "valid"]})
        result = clean_data(df)
        assert result["a"].isna().sum() == 3
        assert result["a"].dropna().iloc[0] == "valid"

    def test_replaces_numeric_sentinel_with_na(self):
        df = pd.DataFrame({"a": [1, -999, 3]})
        result = clean_data(df)
        assert result["a"].isna().sum() == 1

    def test_idempotent(self):
        df = pd.DataFrame({"Name": [1, 2], "Age": [3, 4]})
        first_pass = clean_data(df)
        second_pass = clean_data(first_pass)
        pd.testing.assert_frame_equal(first_pass, second_pass)

    def test_empty_dataframe_returns_empty(self):
        df = pd.DataFrame({"a": [], "b": []})
        result = clean_data(df)
        assert result.empty

    def test_no_object_columns_still_works(self):
        df = pd.DataFrame({"x": [1, 2, 3], "y": [4.0, 5.0, 6.0]})
        result = clean_data(df)
        assert len(result) == 3
