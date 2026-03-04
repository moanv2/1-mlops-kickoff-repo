"""
Module: Data Loading
--------------------
Role: Load the raw telecom churn dataset with validation and logging.
Usage: from src.load_data import load_data
Written by: Diego Alfaro (if you have any questions ask me)
"""

import logging
from pathlib import Path

import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

RAW_DATA_PATH = Path("data/raw/telecom_churn.csv")


def load_csv(path: Path) -> pd.DataFrame:
    """Centralized CSV loader with error handling (Steps 4 & 5)."""
    try:
        df = pd.read_csv(path)
    except Exception as e:
        raise RuntimeError(
            f"Failed to parse CSV at '{path}'.\n"
            f"Check that the file is a valid CSV and not corrupted.\n"
            f"Original error: {e}"
        ) from e
    return df


def load_data(path: Path = RAW_DATA_PATH) -> pd.DataFrame:
    """Load and validate the raw dataset.

    Steps:
        1. Verify the file exists
        2. Validate the path points to a file
        3. Load CSV with centralized parser
        4. Guard against empty data
        5. Log incoming volume
    """
    # Step 1: Enforce file dependency
    if not path.exists():
        raise FileNotFoundError(
            f"Raw dataset not found at '{path}'.\n"
            f"Please place 'telecom_churn.csv' in the data/raw/ folder."
        )

    # Step 2: Validate path type
    if not path.is_file():
        raise ValueError(
            f"Expected a file but got a directory: '{path}'.\n"
            f"Check your RAW_DATA_PATH configuration."
        )

    # Steps 3 & 5: Centralized CSV parsing with error translation
    df = load_csv(path)

    # Step 4: Guard against empty data
    if df.empty:
        raise ValueError(
            f"Loaded file '{path}' contains no rows.\n"
            f"The upstream data export may have failed."
        )

    # Step 6: Log incoming volume
    logger.info(f"Loaded '{path}': {df.shape[0]} rows, {df.shape[1]} columns")

    return df


if __name__ == "__main__":
    df = load_data()
    print(df.head())
