"""Tests for src.validate module."""

import pandas as pd
import pytest

from src.validate import validate_dataframe


class TestValidateDataframe:
    """Tests for validate_dataframe."""

    def test_valid_dataframe_passes(self):
        df = pd.DataFrame({"a": [1, 2], "b": [3, 4]})
        assert validate_dataframe(df, ["a", "b"]) is True

    def test_none_raises(self):
        with pytest.raises(ValueError, match="df is None"):
            validate_dataframe(None, ["a"])

    def test_empty_dataframe_raises(self):
        df = pd.DataFrame({"a": []})
        with pytest.raises(ValueError, match="DataFrame is empty"):
            validate_dataframe(df, ["a"])

    def test_missing_columns_raises(self):
        df = pd.DataFrame({"a": [1]})
        with pytest.raises(ValueError, match="Missing required columns"):
            validate_dataframe(df, ["a", "b"])

    def test_null_values_in_required_column_raises(self):
        df = pd.DataFrame({"a": [1, None], "b": [3, 4]})
        with pytest.raises(ValueError, match="contains missing values"):
            validate_dataframe(df, ["a", "b"])

    def test_duplicate_rows_raises(self):
        df = pd.DataFrame({"a": [1, 1], "b": [2, 2]})
        with pytest.raises(ValueError, match="duplicate row"):
            validate_dataframe(df, ["a", "b"])

    def test_negative_telecom_column_raises(self):
        df = pd.DataFrame({
            "AccountWeeks": [10, -1],
            "DataUsage": [1, 2],
            "CustServCalls": [0, 1],
            "DayMins": [100, 200],
            "DayCalls": [5, 6],
            "MonthlyCharge": [50, 60],
            "OverageFee": [5, 10],
            "RoamMins": [3, 4],
        })
        with pytest.raises(ValueError, match="contains negative values"):
            validate_dataframe(df, ["AccountWeeks"])

    def test_invalid_binary_column_raises(self):
        df = pd.DataFrame({
            "Churn": [0, 2],
            "ContractRenewal": [1, 0],
            "DataPlan": [0, 1],
        })
        with pytest.raises(ValueError, match="values outside"):
            validate_dataframe(df, ["Churn"])
