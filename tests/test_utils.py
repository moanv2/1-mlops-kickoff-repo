"""Tests for src.utils module."""

import pathlib

import pytest

from src.utils import get_project_root, load_config


class TestGetProjectRoot:
    """Tests for get_project_root()."""

    def test_returns_path(self):
        root = get_project_root()
        assert isinstance(root, pathlib.Path)

    def test_root_contains_src(self):
        root = get_project_root()
        assert (root / "src").is_dir()

    def test_root_contains_config(self):
        root = get_project_root()
        assert (root / "config.yaml").is_file()


class TestLoadConfig:
    """Tests for load_config()."""

    def test_returns_dict(self):
        cfg = load_config()
        assert isinstance(cfg, dict)

    def test_has_data_section(self):
        cfg = load_config()
        assert "data" in cfg

    def test_has_train_section(self):
        cfg = load_config()
        assert "train" in cfg

    def test_data_raw_path_is_string(self):
        cfg = load_config()
        assert isinstance(cfg["data"]["raw"], str)

    def test_train_seed_is_int(self):
        cfg = load_config()
        assert isinstance(cfg["train"]["seed"], int)

    def test_train_test_size_is_float(self):
        cfg = load_config()
        assert isinstance(cfg["train"]["test_size"], float)

    def test_raises_on_missing_file(self, tmp_path):
        with pytest.raises(FileNotFoundError, match="Config file not found"):
            load_config(tmp_path / "nonexistent.yaml")

    def test_explicit_path(self, tmp_path):
        cfg_file = tmp_path / "custom.yaml"
        cfg_file.write_text("foo: bar\n", encoding="utf-8")
        cfg = load_config(cfg_file)
        assert cfg == {"foo": "bar"}
