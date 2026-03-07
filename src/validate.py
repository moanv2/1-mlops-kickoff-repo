"""
Educational Goal:
- Why this module exists in an MLOps system: Data validation acts as a quality gate between ingestion and training (GIGO).
- Responsibility (separation of concerns): Check schema/quality only. Do NOT clean, transform, or load data.
- Pipeline contract (inputs and outputs): Receives a DataFrame + required columns. Returns True or raises ValueError (fail-fast).

TODO: Replace print statements with standard library logging in a later session
TODO: Any temporary or hardcoded variable or parameter will be imported from config.yml in a later session
"""

import pandas as pd


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
    print("[validate.validate_dataframe] Starting data validation checks...")  # TODO: replace with logging later

    if df is None:
        raise ValueError("Validation failed: df is None. Upstream step did not return a DataFrame.")

    if df.empty:
        raise ValueError(
            "Validation failed: The DataFrame is empty. Check your load_data and clean_data steps."
        )
    print(f"[validate.validate_dataframe] DataFrame shape: {df.shape}")  # TODO: replace with logging later

    missing_cols = [col for col in required_columns if col not in df.columns]
    if missing_cols:
        raise ValueError(
            f"Validation failed: Missing required columns: {missing_cols}. "
            f"Columns present: {df.columns.tolist()}"
        )
    print("[validate.validate_dataframe] All required columns are present.")  # TODO: replace with logging later

    # --------------------------------------------------------
    # START STUDENT CODE
    # --------------------------------------------------------
    # Student checks (kept), but guarded to avoid breaking other datasets/scaffolding.

    # Check 1: Missing values (simple + safer: only required columns)
    for col in required_columns:
        if df[col].isnull().any():
            raise ValueError(f"Validation failed: Required column '{col}' contains missing values.")
    print("[validate.validate_dataframe] No missing values in required columns.")  # TODO: replace with logging later

    # Check 2: No duplicate rows
    n_duplicates = int(df.duplicated().sum())
    if n_duplicates > 0:
        raise ValueError(f"Validation failed: Dataset contains {n_duplicates} duplicate row(s).")
    print("[validate.validate_dataframe] No duplicate rows found.")  # TODO: replace with logging later

    # Telecom-specific checks (run only if those columns exist)
    non_negative_cols = [
        "AccountWeeks", "DataUsage", "CustServCalls",
        "DayMins", "DayCalls", "MonthlyCharge", "OverageFee", "RoamMins",
    ]
    if set(non_negative_cols).issubset(df.columns):
        for col in non_negative_cols:
            if (df[col] < 0).any():
                raise ValueError(f"Validation failed: Column '{col}' contains negative values.")
        print("[validate.validate_dataframe] Telecom numeric columns are non-negative.")  # TODO: replace with logging later

    binary_cols = ["Churn", "ContractRenewal", "DataPlan"]
    if set(binary_cols).issubset(df.columns):
        for col in binary_cols:
            invalid = df.loc[~df[col].isin([0, 1]), col]
            if not invalid.empty:
                raise ValueError(
                    f"Validation failed: Column '{col}' contains values outside {{0, 1}}: "
                    f"{invalid.unique().tolist()}"
                )
        print("[validate.validate_dataframe] Telecom binary columns contain only 0/1.")  # TODO: replace with logging later
    # --------------------------------------------------------
    # END STUDENT CODE
    # --------------------------------------------------------

    print("[validate.validate_dataframe] All validation checks passed.")  # TODO: replace with logging later
    return True