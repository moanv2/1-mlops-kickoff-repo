"""
Educational Goal:

Why this module exists in an MLOps system: Evaluation is the gate that decides whether a trained
model is good enough to move forward in the pipeline (to registration, deployment, or A/B testing).
Without a dedicated, isolated evaluation step, it is easy to accidentally leak information from the
test set into training, producing optimistic metrics that do not reflect real-world performance.

Responsibility (separation of concerns): This module measures how well a fitted model performs on
held-out data AND produces diagnostic plots saved to reports/figures/. It does NOT load data and
does NOT train or modify the model — those concerns belong to ingest.py and train.py respectively.

Pipeline contract (inputs and outputs): Receives a fitted sklearn Pipeline object, a feature
DataFrame (X_test), a target Series (y_test), and a string that declares the problem type.
Returns a single float — the primary evaluation metric — so the caller (src/main.py) can make
promotion decisions with a simple numerical comparison. As a side-effect, saves a diagnostic plot
to reports/figures/ and prints a full metrics dictionary to stdout.

TODO: Replace print statements with standard library logging in a later session
TODO: Any temporary or hardcoded variable or parameter will be imported from config.yml in a later session
"""
"""
evaluate.py

Fixes applied (per project priorities):
- Use config.yaml (if present) instead of hardcoded constants for reports directory
- Replace print() with logging
- Keep REPORTS_DIR global so existing tests can monkeypatch it
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict

import pandas as pd
import yaml

LOGGER = logging.getLogger(__name__)

# Kept for backward compatibility + tests that monkeypatch this
REPORTS_DIR = Path("reports") / "figures"


def _load_config(config_path: str = "config.yaml") -> Dict[str, Any]:
    path = Path(config_path)
    if not path.exists():
        LOGGER.info("Config file not found at %s. Using defaults.", config_path)
        return {}
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def evaluate_model(
    model,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    problem_type: str,
    config_path: str = "config.yaml",
) -> float:
    """
    Evaluate a fitted model on a held-out split.

    Returns:
        float: primary metric
              - regression: RMSE (lower is better)
              - classification: weighted F1 (higher is better)
    Side effects:
        Saves a diagnostic plot to reports/figures (or config override).
    """
    cfg = _load_config(config_path)
    eval_cfg = cfg.get("evaluate", cfg)

    # Allow config override, else use the module constant (supports monkeypatch in tests)
    reports_dir = Path(eval_cfg.get("reports_dir", str(REPORTS_DIR)))
    reports_dir.mkdir(parents=True, exist_ok=True)

    problem_type = problem_type.strip().lower()
    LOGGER.info("Starting evaluation: problem_type=%s", problem_type)

    # Guardrails
    if not hasattr(model, "predict"):
        raise ValueError("Model must implement .predict().")
    if not isinstance(X_test, pd.DataFrame):
        raise ValueError("X_test must be a pandas DataFrame.")
    if X_test.empty:
        raise ValueError("X_test is empty.")
    if len(X_test) != len(y_test):
        raise ValueError(
            f"Length mismatch: X_test has {len(X_test)} rows but y_test has {len(y_test)} entries."
        )

    y_pred = model.predict(X_test)

    # Plotting (non-interactive backend)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if problem_type == "regression":
        from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

        rmse = float(mean_squared_error(y_test, y_pred, squared=False))
        mae = float(mean_absolute_error(y_test, y_pred))
        r2 = float(r2_score(y_test, y_pred))
        LOGGER.info("Regression metrics: rmse=%.6f mae=%.6f r2=%.6f", rmse, mae, r2)

        residuals = (
            pd.Series(y_test).astype(float).to_numpy()
            - pd.Series(y_pred).astype(float).to_numpy()
        )

        fig, ax = plt.subplots()
        ax.scatter(y_pred, residuals, alpha=0.4)
        ax.axhline(0, linewidth=1, linestyle="--")
        ax.set_xlabel("Predicted value")
        ax.set_ylabel("Residual (actual − predicted)")
        ax.set_title("Residual Plot")
        fig.tight_layout()

        plot_path = reports_dir / "residual_plot.png"
        fig.savefig(plot_path)
        plt.close(fig)
        LOGGER.info("Saved residual plot to %s", plot_path)

        return rmse

    if problem_type == "classification":
        from sklearn.metrics import classification_report, confusion_matrix, f1_score

        f1w = float(f1_score(y_test, y_pred, average="weighted"))
        LOGGER.info("Classification metric: f1_weighted=%.6f", f1w)
        LOGGER.info("Classification report:\n%s", classification_report(y_test, y_pred))

        cm = confusion_matrix(y_test, y_pred)

        fig, ax = plt.subplots()
        im = ax.imshow(cm, interpolation="nearest", cmap="Blues")
        fig.colorbar(im, ax=ax)
        ax.set_xlabel("Predicted label")
        ax.set_ylabel("True label")
        ax.set_title("Confusion Matrix")
        fig.tight_layout()

        plot_path = reports_dir / "confusion_matrix.png"
        fig.savefig(plot_path)
        plt.close(fig)
        LOGGER.info("Saved confusion matrix to %s", plot_path)

        return f1w

    raise ValueError(
        f"Unknown problem_type='{problem_type}'. Expected 'regression' or 'classification'."
    )