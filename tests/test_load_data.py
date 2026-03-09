"""Tests for src.load_data module."""

import pandas as pd
import pytest

from src.load_data import load_csv, load_data


class TestLoadCsv:
    """Tests for the load_csv helper."""

    def test_loads_valid_csv(self, tmp_path):
        csv_file = tmp_path / "valid.csv"
        csv_file.write_text("a,b\n1,2\n3,4\n")
        df = load_csv(csv_file)
        assert isinstance(df, pd.DataFrame)
        assert df.shape == (2, 2)

    def test_raises_on_invalid_file(self, tmp_path):
        bad_file = tmp_path / "bad.csv"
        bad_file.write_bytes(b"\x00\x01\x02")
        with pytest.raises(RuntimeError, match="Failed to parse CSV"):
            load_csv(bad_file)


class TestLoadData:
    """Tests for the load_data entry point."""

    def test_loads_valid_dataset(self, tmp_path):
        csv_file = tmp_path / "data.csv"
        csv_file.write_text("col1,col2\n1,2\n3,4\n")
        df = load_data(csv_file)
        assert isinstance(df, pd.DataFrame)
        assert not df.empty

    def test_raises_file_not_found(self, tmp_path):
        missing = tmp_path / "missing.csv"
        with pytest.raises(FileNotFoundError, match="Raw dataset not found"):
            load_data(missing)

    def test_raises_on_directory(self, tmp_path):
        with pytest.raises(ValueError, match="Expected a file but got a directory"):
            load_data(tmp_path)

    def test_raises_on_empty_csv(self, tmp_path):
        csv_file = tmp_path / "empty.csv"
        csv_file.write_text("col1,col2\n")
        with pytest.raises(ValueError, match="contains no rows"):
            load_data(csv_file)
