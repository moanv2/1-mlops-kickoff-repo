"""Tests for src.utils module."""

import logging
import pathlib

import pytest

from src.utils import get_project_root, load_config, setup_logger


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


class TestSetupLogger:
    """Tests for setup_logger()."""

    def test_returns_logger(self):
        logger = setup_logger("test_utils_logger")
        assert isinstance(logger, logging.Logger)

    def test_logger_has_handler(self):
        logger = setup_logger("test_handler_check")
        assert len(logger.handlers) >= 1

    def test_default_level_is_info(self):
        logger = setup_logger("test_level_check")
        assert logger.level == logging.INFO

    def test_custom_level(self):
        logger = setup_logger("test_debug_level", level=logging.DEBUG)
        assert logger.level == logging.DEBUG

    def test_same_name_returns_same_logger(self):
        a = setup_logger("test_same_name")
        b = setup_logger("test_same_name")
        assert a is b

    def test_no_duplicate_handlers(self):
        name = "test_no_dup_handlers"
        setup_logger(name)
        setup_logger(name)
        logger = logging.getLogger(name)
        assert len(logger.handlers) == 1
