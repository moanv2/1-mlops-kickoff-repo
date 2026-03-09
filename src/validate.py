"""
Educational Goal:
- Why this module exists in an MLOps system: Data validation acts as a quality gate between ingestion and training (GIGO).
- Responsibility (separation of concerns): Check schema/quality only. Do NOT clean, transform, or load data.
- Pipeline contract (inputs and outputs): Receives a DataFrame + required columns. Returns True or raises ValueError (fail-fast).

TODO: Any temporary or hardcoded variable or parameter will be imported from config.yml in a later session
"""

import logging
import pathlib
import pandas as pd
import yaml

logger = logging.getLogger(__name__)

_CONFIG_PATH = pathlib.Path(__file__).parent.parent / "config.yaml"

with open(_CONFIG_PATH, "r") as _f:
    _config = yaml.safe_load(_f)

_validate_cfg = _config.get("validation", {})


def validate_dataframe(df: pd.DataFrame, required_columns: list) -> bool:
    """
    Inputs:
    - df: A pandas DataFrame to validate (typically the cleaned DataFrame).
    - required_columns: A list of column name strings that must be present in df.
    Outputs:
    - True if all validation checks pass.
    - Raises ValueError immediately if any check fails (fail-fast).
    Why this contract matters for reliable ML delivery:
    - Early, clear failures reduce debugging time and prevent silent corruption of metrics/artifacts.
    """
    logger.info("Starting data validation checks...")

    if df is None:
        raise ValueError("Validation failed: df is None. Upstream step did not return a DataFrame.")

    if df.empty:
        raise ValueError(
            "Validation failed: The DataFrame is empty. Check your load_data and clean_data steps."
        )
    logger.info("DataFrame shape: %s", df.shape)

    missing_cols = [col for col in required_columns if col not in df.columns]
    if missing_cols:
        raise ValueError(
            f"Validation failed: Missing required columns: {missing_cols}. "
            f"Columns present: {df.columns.tolist()}"
        )
    logger.info("All required columns are present.")

    # --------------------------------------------------------
    # START STUDENT CODE
    # --------------------------------------------------------
    # Student checks (kept), but guarded to avoid breaking other datasets/scaffolding.

    # Check 1: Missing values (simple + safer: only required columns)
    for col in required_columns:
        if df[col].isnull().any():
            raise ValueError(f"Validation failed: Required column '{col}' contains missing values.")
    logger.info("No missing values in required columns.")

    # Check 2: No duplicate rows
    n_duplicates = int(df.duplicated().sum())
    if n_duplicates > 0:
        raise ValueError(f"Validation failed: Dataset contains {n_duplicates} duplicate row(s).")
    logger.info("No duplicate rows found.")

    # Telecom-specific checks (run only if those columns exist)
    non_negative_cols = _validate_cfg.get("non_negative_cols", [])
    if non_negative_cols and set(non_negative_cols).issubset(df.columns):
        for col in non_negative_cols:
            if (df[col] < 0).any():
                raise ValueError(f"Validation failed: Column '{col}' contains negative values.")
        logger.info("Telecom numeric columns are non-negative.")

    binary_cols = _validate_cfg.get("binary_cols", [])
    if binary_cols and set(binary_cols).issubset(df.columns):
        for col in binary_cols:
            invalid = df.loc[~df[col].isin([0, 1]), col]
            if not invalid.empty:
                raise ValueError(
                    f"Validation failed: Column '{col}' contains values outside {{0, 1}}: "
                    f"{invalid.unique().tolist()}"
                )
        logger.info("Telecom binary columns contain only 0/1.")
    # --------------------------------------------------------
    # END STUDENT CODE
    # --------------------------------------------------------

    logger.info("All validation checks passed.")
    return True