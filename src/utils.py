"""
Module: Shared Utilities
------------------------
Role: Provide common helpers used across pipeline modules.
    - load_config(): read config.yaml into a dict
    - setup_logger(): consistent logging for every module
    - get_project_root(): resolve the repo root path
"""

import logging
import pathlib

import yaml


def get_project_root() -> pathlib.Path:
    """Return the repository root (parent of src/)."""
    return pathlib.Path(__file__).resolve().parent.parent


def load_config(path: str | pathlib.Path | None = None) -> dict:
    """Load config.yaml and return its contents as a dict.

    Parameters
    ----------
    path : str or Path, optional
        Explicit path to the YAML config file.
        Defaults to ``<project_root>/config.yaml``.

    Returns
    -------
    dict
        Parsed configuration.

    Raises
    ------
    FileNotFoundError
        If the config file does not exist at the resolved path.
    """
    if path is None:
        path = get_project_root() / "config.yaml"
    path = pathlib.Path(path)

    if not path.exists():
        raise FileNotFoundError(
            f"Config file not found at '{path}'. "
            "Make sure config.yaml is in the project root."
        )

    with open(path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    return config or {}


def setup_logger(name: str, level: int = logging.INFO) -> logging.Logger:
    """Return a logger with a consistent format.

    Calling this multiple times with the same *name* returns the same
    logger instance (standard ``logging`` behaviour), so it is safe
    to call from every module's top level.

    Parameters
    ----------
    name : str
        Logger name — pass ``__name__`` from the calling module.
    level : int, optional
        Logging level (default ``logging.INFO``).
    """
    logger = logging.getLogger(name)

    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)

    logger.setLevel(level)
    return logger
