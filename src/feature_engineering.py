"""
Role: Doing feature engineering
"""



from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional

import pandas as pd


@dataclass
class FeatureConfig:
    """Configuration for feature engineering."""
    target_col: Optional[str] = None
    drop_cols: tuple[str, ...] = ()
    date_cols: tuple[str, ...] = ()
    categorical_cols: tuple[str, ...] = ()
    numeric_cols: tuple[str, ...] = ()


def _parse_dates(df: pd.DataFrame, date_cols: Iterable[str]) -> pd.DataFrame:
    df = df.copy()
    for col in date_cols:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors="coerce")
            df[f"{col}_year"] = df[col].dt.year
            df[f"{col}_month"] = df[col].dt.month
            df[f"{col}_day"] = df[col].dt.day
            df[f"{col}_dow"] = df[col].dt.dayofweek
    return df


def _encode_categoricals(df: pd.DataFrame, categorical_cols: Iterable[str]) -> pd.DataFrame:
    """One-hot encode categoricals with safe handling for missing cols."""
    df = df.copy()
    existing = [c for c in categorical_cols if c in df.columns]
    if not existing:
        return df
    return pd.get_dummies(df, columns=existing, dummy_na=True)


def _clean_numerics(df: pd.DataFrame, numeric_cols: Iterable[str]) -> pd.DataFrame:
    df = df.copy()
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
            # Simple example: missing indicator + fill
            df[f"{col}_is_missing"] = df[col].isna().astype(int)
            df[col] = df[col].fillna(df[col].median())
    return df


def build_features(df: pd.DataFrame, cfg: FeatureConfig) -> pd.DataFrame:
    """
    Main feature engineering pipeline.

    Returns a new DataFrame with engineered features.
    """
    out = df.copy()

    # Drop columns you don't want as features (IDs, leakage, etc.)
    for c in cfg.drop_cols:
        if c in out.columns:
            out = out.drop(columns=[c])

    out = _parse_dates(out, cfg.date_cols)
    out = _clean_numerics(out, cfg.numeric_cols)
    out = _encode_categoricals(out, cfg.categorical_cols)

    # If target exists, keep it at the end (optional)
    if cfg.target_col and cfg.target_col in df.columns:
        out[cfg.target_col] = df[cfg.target_col]

    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Run feature engineering and save output.")
    parser.add_argument("--input", required=True, help="Path to input CSV")
    parser.add_argument("--output", required=True, help="Path to output CSV")
    args = parser.parse_args()

    in_path = Path(args.input)
    out_path = Path(args.output)

    df = pd.read_csv(in_path)

    # TODO: customize these for your dataset
    cfg = FeatureConfig(
        target_col=None,  # set to your target column if you want to keep it
        drop_cols=("id",),
        date_cols=("date",),
        categorical_cols=("country", "segment"),
        numeric_cols=("age", "income"),
    )

    features = build_features(df, cfg)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    features.to_csv(out_path, index=False)
    print(f"Saved features to: {out_path} (shape={features.shape})")


if __name__ == "__main__":
    main()