"""
Module: Shared Utilities
------------------------
Role: Provide common helpers used across pipeline modules.
    - load_config(): read config.yaml into a dict
    - get_project_root(): resolve the repo root path
"""

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
