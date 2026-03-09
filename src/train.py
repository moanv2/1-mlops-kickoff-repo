"""
train.py

Fixes applied (per project priorities):
- Load defaults from config.yaml instead of hardcoded constants (with safe fallback)
- Replace print() with logging
- Implement 3-way split: train/validation/test
- Keep Pipeline step names: ["preprocess", "model"]
- Save fitted pipeline artifact to disk (create directory if missing)

Returns:
    fitted_pipeline, X_val, y_val, X_test, y_test
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import joblib
import pandas as pd
import yaml
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

LOGGER = logging.getLogger(__name__)


def _load_config(config_path: str = "config.yaml") -> Dict[str, Any]:
    """Load config.yaml if present; otherwise return {} (keeps module runnable)."""
    path = Path(config_path)
    if not path.exists():
        LOGGER.info("Config file not found at %s. Using function defaults.", config_path)
        return {}
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def train_model(
    X: pd.DataFrame,
    y: pd.Series,
    preprocessor,
    problem_type: str,
    model_path: Optional[str] = None,
    test_size: Optional[float] = None,
    val_size: Optional[float] = None,
    random_state: Optional[int] = None,
    config_path: str = "config.yaml",
) -> Tuple[Pipeline, pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
    """
    Train a sklearn Pipeline (preprocessor + estimator) with a 3-way split.

    Parameters
    ----------
    X : pd.DataFrame
        Full feature matrix.
    y : pd.Series | single-column pd.DataFrame
        Full target vector aligned with X.
    preprocessor : Transformer
        ColumnTransformer (or similar) used in the pipeline as "preprocess".
    problem_type : str
        "regression" or "classification".
    model_path : str | None
        Where to save the fitted pipeline artifact (joblib).
    test_size : float | None
        Fraction held out for final test split (from full dataset).
    val_size : float | None
        Fraction held out for validation split (from full dataset).
    random_state : int | None
        Seed for reproducibility.
    config_path : str
        Path to config.yaml to source defaults.

    Returns
    -------
    (fitted_pipeline, X_val, y_val, X_test, y_test)
    """
    cfg = _load_config(config_path)
    train_cfg = cfg.get("train", cfg)  # tolerate either train: block or top-level keys

    model_path = model_path or train_cfg.get("model_path", "models/model.pkl")
    test_size = float(test_size if test_size is not None else train_cfg.get("test_size", 0.2))
    val_size = float(val_size if val_size is not None else train_cfg.get("val_size", 0.2))
    random_state = int(random_state if random_state is not None else train_cfg.get("random_state", 42))

    problem_type_normalized = problem_type.strip().lower()
    LOGGER.info("Starting training: problem_type=%s", problem_type_normalized)

    # -----------------------------
    # Guardrails
    # -----------------------------
    if not isinstance(X, pd.DataFrame):
        raise ValueError("X must be a pandas DataFrame.")
    if isinstance(y, pd.DataFrame):
        if y.shape[1] != 1:
            raise ValueError("y DataFrame must have exactly one column.")
        y = y.iloc[:, 0]
    if not isinstance(y, pd.Series):
        raise ValueError("y must be a pandas Series (or single-column DataFrame).")

    if X.empty:
        raise ValueError("X is empty. Cannot train on zero samples.")
    if len(X) != len(y):
        raise ValueError(f"Shape mismatch: X has {len(X)} rows but y has {len(y)} entries.")

    if not (0.0 < test_size < 1.0) or not (0.0 < val_size < 1.0):
        raise ValueError("test_size and val_size must be floats in (0, 1).")
    if test_size + val_size >= 1.0:
        raise ValueError("test_size + val_size must be < 1.0.")

    # -----------------------------
    # 3-way split: train/val/test
    # -----------------------------
    stratify = y if problem_type_normalized == "classification" else None

    LOGGER.info(
        "Splitting data (3-way): test_size=%.3f val_size=%.3f random_state=%d",
        test_size,
        val_size,
        random_state,
    )

    # Split off test
    X_temp, X_test, y_temp, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=stratify,
    )

    # Split remaining into train/val; convert val_size to fraction of remaining
    val_fraction_of_temp = val_size / (1.0 - test_size)
    stratify_temp = y_temp if problem_type_normalized == "classification" else None

    X_train, X_val, y_train, y_val = train_test_split(
        X_temp,
        y_temp,
        test_size=val_fraction_of_temp,
        random_state=random_state,
        stratify=stratify_temp,
    )

    LOGGER.info("Split sizes: train=%d val=%d test=%d", len(X_train), len(X_val), len(X_test))

    # -----------------------------
    # Model selection
    # -----------------------------
    if problem_type_normalized == "regression":
        estimator = Ridge(random_state=random_state)
    elif problem_type_normalized == "classification":
        estimator = LogisticRegression(
            solver="liblinear",
            max_iter=500,
            random_state=random_state,
        )
    else:
        raise ValueError("Unsupported problem_type. Expected 'regression' or 'classification'.")

    pipeline = Pipeline(
        steps=[
            ("preprocess", preprocessor),
            ("model", estimator),
        ]
    )

    LOGGER.info("Fitting pipeline on training split only...")
    fitted_pipeline = pipeline.fit(X_train, y_train)

    # -----------------------------
    # Save artifact
    # -----------------------------
    os.makedirs(os.path.dirname(model_path) or ".", exist_ok=True)
    joblib.dump(fitted_pipeline, model_path)
    LOGGER.info("Saved model artifact to %s", model_path)

    return fitted_pipeline, X_val, y_val, X_test, y_test