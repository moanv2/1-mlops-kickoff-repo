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

import pathlib

import pandas as pd

try:
    from sklearn.metrics import f1_score, root_mean_squared_error as _rmse_fn

    def _compute_rmse(y_true, y_pred):
        return float(_rmse_fn(y_true, y_pred))

except ImportError:
    import math
    from sklearn.metrics import f1_score, mean_squared_error as _mse_fn

    def _compute_rmse(y_true, y_pred):
        return float(math.sqrt(_mse_fn(y_true, y_pred)))


# TODO: import from config.yml in a later session
REPORTS_DIR = pathlib.Path("reports") / "figures"


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def evaluate_model(
    model,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    problem_type: str,
) -> float:
    """
    Inputs:

        model        — a fitted sklearn Pipeline (or any estimator) that exposes .predict().
        X_test       — pd.DataFrame of feature columns, never seen during training.
        y_test       — pd.Series of ground-truth target values aligned with X_test.
        problem_type — str, either "regression" or "classification".

    Outputs:

        float — the primary evaluation metric:
                  "regression"     → RMSE  (lower is better)
                  "classification" → weighted F1 score (higher is better, range [0, 1])
        Side-effects:
                  Prints a full metrics dictionary to stdout.
                  Saves a diagnostic plot to reports/figures/.

    Why this contract matters for reliable ML delivery:

        Returning a single, well-defined float lets the orchestrator (main.py) make objective,
        automated promotion decisions (e.g., "deploy only if RMSE < threshold"). The side-effect
        plots give data scientists the visual diagnostics they need to understand model behaviour
        without polluting the pipeline's decision-making logic with display code.
    """
    print("[evaluate.evaluate_model] Starting model evaluation...")  # TODO: replace with logging later

    # ------------------------------------------------------------------
    # Fail-fast guardrails — catch problems before sklearn sees bad data
    # ------------------------------------------------------------------

    if not hasattr(model, "predict"):
        raise ValueError(
            "Model artifact contract violation: the object passed as 'model' does not expose "
            "a .predict() method. Pass a fitted sklearn Pipeline or estimator."
        )

    if X_test.empty:
        raise ValueError(
            "Guardrail triggered: X_test is empty. Evaluation requires at least one sample."
        )

    if len(X_test) != len(y_test):
        raise ValueError(
            f"Guardrail triggered: X_test has {len(X_test)} rows but y_test has {len(y_test)} "
            "entries. They must be the same length."
        )

    # ------------------------------------------------------------------
    # Predict on untouched test data — model is never modified here
    # ------------------------------------------------------------------

    print("[evaluate.evaluate_model] Running model.predict() on X_test...")  # TODO: replace with logging later
    y_pred = model.predict(X_test)

    # ------------------------------------------------------------------
    # --------------------------------------------------------
    # START STUDENT CODE
    # --------------------------------------------------------
    # TODO_STUDENT: Paste your notebook logic here to replace or extend the baseline
    # Why: Every dataset and business context demands different diagnostics. A regression
    #      problem for house prices needs residual plots and MAE; a fraud-detection classifier
    #      needs a confusion matrix and precision-recall curves. The baseline below gives you
    #      one working plot per problem type — extend or swap it for your use case.
    # Examples:
    # 1. Regression — swap the residual scatter for a prediction-vs-actual line plot:
    #       ax.plot(y_test, y_pred, "o", alpha=0.4)
    #       ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], "r--")
    # 2. Classification — add a normalised confusion matrix so class imbalance doesn't hide errors:
    #       cm = confusion_matrix(y_test, y_pred, normalize="true")
    # Optional forcing function (leave commented):
    # raise NotImplementedError("Student: You must implement this logic to proceed!")
    # ------------------------------------------------------------------

    import matplotlib  # import here so matplotlib is not a hard top-level dependency
    matplotlib.use("Agg")  # non-interactive backend — safe inside a pipeline
    import matplotlib.pyplot as plt

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    if problem_type == "regression":
        from sklearn.metrics import mean_absolute_error, r2_score

        metrics = {
            "rmse": _compute_rmse(y_test, y_pred),
            "mae": float(mean_absolute_error(y_test, y_pred)),
            "r2": float(r2_score(y_test, y_pred)),
        }
        print(f"[evaluate.evaluate_model] Metrics: {metrics}")  # TODO: replace with logging later

        # Residual plot: shows where predictions are systematically wrong
        residuals = [float(a) - float(p) for a, p in zip(y_test, y_pred)]
        fig, ax = plt.subplots()
        ax.scatter(y_pred, residuals, alpha=0.4)
        ax.axhline(0, color="red", linewidth=1, linestyle="--")
        ax.set_xlabel("Predicted value")
        ax.set_ylabel("Residual  (actual − predicted)")
        ax.set_title("Residual Plot")
        fig.tight_layout()
        plot_path = REPORTS_DIR / "residual_plot.png"
        fig.savefig(plot_path)
        plt.close(fig)
        print(f"[evaluate.evaluate_model] Saved residual plot → {plot_path}")  # TODO: replace with logging later

    elif problem_type == "classification":
        from sklearn.metrics import classification_report, confusion_matrix

        metrics = {
            "f1_weighted": float(f1_score(y_test, y_pred, average="weighted")),
        }
        report = classification_report(y_test, y_pred)
        print(f"[evaluate.evaluate_model] Metrics: {metrics}")  # TODO: replace with logging later
        print(f"[evaluate.evaluate_model] Classification Report:\n{report}")  # TODO: replace with logging later

        # Confusion matrix plot: rows = true labels, columns = predicted labels
        cm = confusion_matrix(y_test, y_pred)
        fig, ax = plt.subplots()
        im = ax.imshow(cm, interpolation="nearest", cmap="Blues")
        fig.colorbar(im, ax=ax)
        ax.set_xlabel("Predicted label")
        ax.set_ylabel("True label")
        ax.set_title("Confusion Matrix")
        fig.tight_layout()
        plot_path = REPORTS_DIR / "confusion_matrix.png"
        fig.savefig(plot_path)
        plt.close(fig)
        print(f"[evaluate.evaluate_model] Saved confusion matrix → {plot_path}")  # TODO: replace with logging later

    # --------------------------------------------------------
    # END STUDENT CODE
    # --------------------------------------------------------
    # ------------------------------------------------------------------

    # ------------------------------------------------------------------
    # Return single primary metric for pipeline promotion decisions
    # ------------------------------------------------------------------

    if problem_type == "regression":
        metric_value = _compute_rmse(y_test, y_pred)
        print(f"[evaluate.evaluate_model] RMSE={metric_value:.4f}")  # TODO: replace with logging later

    elif problem_type == "classification":
        metric_value = float(f1_score(y_test, y_pred, average="weighted"))
        print(f"[evaluate.evaluate_model] F1_weighted={metric_value:.4f}")  # TODO: replace with logging later

    else:
        raise ValueError(
            f"Unknown problem_type='{problem_type}'. Expected 'regression' or 'classification'."
        )

    print("[evaluate.evaluate_model] Evaluation complete.")  # TODO: replace with logging later
    return metric_value
