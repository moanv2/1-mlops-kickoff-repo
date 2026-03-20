"""
Module: Centralized Logger
--------------------------
Role: Provide a single logging setup used by every module in the pipeline.
      Writes to both console (stderr) and a local log file.
Usage: from src.logger import get_logger
       logger = get_logger(__name__)
"""

import logging
import sys
from pathlib import Path

_LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
_INITIALIZED = False


def setup_logging(
    log_file: str = "logs/pipeline.log",
    level: str = "INFO",
) -> None:
    """Configure the root logger with console and file handlers.

    Safe to call multiple times — only the first call takes effect.
    """
    global _INITIALIZED
    if _INITIALIZED:
        return
    _INITIALIZED = True

    log_level = getattr(logging, level.upper(), logging.INFO)
    formatter = logging.Formatter(_LOG_FORMAT)

    root = logging.getLogger()
    root.setLevel(log_level)

    # Console handler (stderr)
    console_handler = logging.StreamHandler(sys.stderr)
    console_handler.setLevel(log_level)
    console_handler.setFormatter(formatter)
    root.addHandler(console_handler)

    # File handler
    log_path = Path(log_file)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    file_handler = logging.FileHandler(log_path, encoding="utf-8")
    file_handler.setLevel(log_level)
    file_handler.setFormatter(formatter)
    root.addHandler(file_handler)


def get_logger(name: str) -> logging.Logger:
    """Return a named logger. Call setup_logging() first from main.py."""
    return logging.getLogger(name)
