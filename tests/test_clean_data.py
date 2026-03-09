"""Tests for src.clean_data module."""

import pandas as pd
import pytest

from src.clean_data import clean_data


class TestCleanDataColumnStandardization:
    """Tests for column name standardization."""

    def test_columns_lowercased(self):
        df = pd.DataFrame({"Name": [1], "AGE": [2]})
        result = clean_data(df)
        assert list(result.columns) == ["name", "age"]

    def test_lowercases_real_column_names(self):
        df = pd.DataFrame({"AccountWeeks": [10], "DataUsage": [5]})
        result = clean_data(df)
        assert list(result.columns) == ["accountweeks", "datausage"]

    def test_columns_spaces_replaced_with_underscores(self):
        df = pd.DataFrame({"First Name": [1], "Last Name": [2]})
        result = clean_data(df)
        assert "first_name" in result.columns
        assert "last_name" in result.columns

    def test_columns_whitespace_stripped(self):
        df = pd.DataFrame({" Name ": [1], "Age ": [2]})
        result = clean_data(df)
        assert "name" in result.columns
        assert "age" in result.columns


class TestCleanDataWhitespace:
    """Tests for string whitespace trimming."""

    def test_strips_whitespace_from_object_columns(self):
        df = pd.DataFrame({"city": ["  Madrid ", " London"]})
        result = clean_data(df)
        assert result["city"].tolist() == ["Madrid", "London"]

    def test_strips_whitespace_lowercase_values(self):
        df = pd.DataFrame({"name": ["  alice  ", " bob"]})
        result = clean_data(df)
        assert result["name"].iloc[0] == "alice"
        assert result["name"].iloc[1] == "bob"

    def test_numeric_columns_unaffected(self):
        df = pd.DataFrame({"value": [1.5, 2.5]})
        result = clean_data(df)
        assert result["value"].tolist() == [1.5, 2.5]

    def test_no_object_columns_still_works(self):
        df = pd.DataFrame({"x": [1, 2, 3], "y": [4.0, 5.0, 6.0]})
        result = clean_data(df)
        assert len(result) == 3


class TestCleanDataDuplicates:
    """Tests for duplicate row removal."""

    def test_removes_exact_duplicate_rows(self):
        df = pd.DataFrame({"a": [1, 1, 2], "b": [3, 3, 4]})
        result = clean_data(df)
        assert len(result) == 2

    def test_keeps_unique_rows(self):
        df = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
        result = clean_data(df)
        assert len(result) == 3


class TestCleanDataMissingValues:
    """Tests for missing value standardization."""

    def test_replaces_na_string_with_pd_na(self):
        df = pd.DataFrame({"col": ["NA", "valid"]})
        result = clean_data(df)
        assert pd.isna(result["col"].iloc[0])
        assert result["col"].iloc[1] == "valid"

    def test_replaces_n_a_string(self):
        df = pd.DataFrame({"col": ["N/A", "ok"]})
        result = clean_data(df)
        assert pd.isna(result["col"].iloc[0])

    def test_replaces_question_mark(self):
        df = pd.DataFrame({"col": ["?", "ok"]})
        result = clean_data(df)
        assert pd.isna(result["col"].iloc[0])

    def test_replaces_null_string(self):
        df = pd.DataFrame({"col": ["null", "ok"]})
        result = clean_data(df)
        assert pd.isna(result["col"].iloc[0])

    def test_replaces_none_string(self):
        df = pd.DataFrame({"col": ["None", "ok"]})
        result = clean_data(df)
        assert pd.isna(result["col"].iloc[0])

    def test_replaces_missing_string(self):
        df = pd.DataFrame({"col": ["missing", "ok"]})
        result = clean_data(df)
        assert pd.isna(result["col"].iloc[0])

    def test_replaces_negative_999(self):
        df = pd.DataFrame({"col": [-999, 42]})
        result = clean_data(df)
        assert pd.isna(result["col"].iloc[0])
        assert result["col"].iloc[1] == 42

    def test_replaces_multiple_sentinels_in_one_column(self):
        df = pd.DataFrame({"a": ["NA", "N/A", "?", "valid"]})
        result = clean_data(df)
        assert result["a"].isna().sum() == 3
        assert result["a"].dropna().iloc[0] == "valid"


class TestCleanDataGeneral:
    """General behavior tests."""

    def test_returns_dataframe(self):
        df = pd.DataFrame({"a": [1, 2]})
        result = clean_data(df)
        assert isinstance(result, pd.DataFrame)

    def test_does_not_mutate_original(self):
        df = pd.DataFrame({"Name": [1, 2]})
        original_columns = list(df.columns)
        clean_data(df)
        assert list(df.columns) == original_columns

    def test_handles_empty_dataframe(self):
        df = pd.DataFrame({"a": []})
        result = clean_data(df)
        assert isinstance(result, pd.DataFrame)
        assert len(result) == 0

    def test_idempotent(self):
        df = pd.DataFrame({"Name": [1, 2], "Age": [3, 4]})
        first_pass = clean_data(df)
        second_pass = clean_data(first_pass)
        pd.testing.assert_frame_equal(first_pass, second_pass)

    def test_mixed_types_and_duplicates(self):
        df = pd.DataFrame({
            "Name": ["  Alice ", "Bob", "  Alice "],
            "Score": [10, 20, 10],
        })
        result = clean_data(df)
        assert "name" in result.columns
        assert len(result) == 2
        assert result["name"].iloc[0] == "Alice"
