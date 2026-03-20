"""
Module: Data Cleaning
---------------------
Role: Preprocessing, missing value imputation, and feature engineering.
Input: pandas.DataFrame (Raw).
Output: pandas.DataFrame (Processed/Clean).
"""
from __future__ import annotations

from typing import Any, Dict, Optional

import pandas as pd

_DEFAULT_SENTINELS = ["NA", "N/A", "", "?", "null", "None", "missing", -999]


def clean_data(df: pd.DataFrame, config: Optional[Dict[str, Any]] = None) -> pd.DataFrame:
    """Clean raw dataset and return a clean DataFrame."""
    df = df.copy()

    # Column name standardization
    df.columns = (
        df.columns.astype("str")
        .str.strip()
        .str.lower()
        .str.replace(" ", "_", regex=False)
    )

    # Trim whitespaces
    obj_cols = df.select_dtypes(include="object").columns
    for col in obj_cols:
        df[col] = df[col].str.strip()

    # Drop exact duplicates rows
    df = df.drop_duplicates()

    # Standardizing missing values (from config or defaults)
    sentinel_values = _DEFAULT_SENTINELS
    if config is not None:
        sentinel_values = config.get("cleaning", {}).get("sentinel_values", _DEFAULT_SENTINELS)
    df = df.replace({val: pd.NA for val in sentinel_values})

    return df
