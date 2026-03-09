"""
Module: Data Cleaning
---------------------
Role: Preprocessing, missing value imputation, and feature engineering.
Input: pandas.DataFrame (Raw).
Output: pandas.DataFrame (Processed/Clean).
"""
from __future__ import annotations

import pandas as pd


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
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

    # Standardizing missing values
    _SENTINEL = ["NA", "N/A", "", "?", "null", "None", "missing", -999]
    df = df.replace({val: pd.NA for val in _SENTINEL})

    return df
