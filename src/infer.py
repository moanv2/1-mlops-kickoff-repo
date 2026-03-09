"""
Module: Inference
-----------------
Role: Provide a deployment-like prediction interface separate from training
    and evaluation.
Responsibility: Run model.predict on new data and return a standardized
    predictions DataFrame.
Pipeline contract: Input fitted model + features; output DataFrame with
    one column 'prediction'.
"""

import pandas as pd

import logging

from src.utils import get_project_root, load_config

logger = logging.getLogger(__name__)


def run_inference(model, X_infer: pd.DataFrame) -> pd.DataFrame:
    """
    Inputs:
    - model: Fitted sklearn Pipeline.
    - X_infer: Features DataFrame for inference.
    Outputs:
    - df_pred: DataFrame with exactly one column named 'prediction', preserving X_infer index.
    Why this contract matters for reliable ML delivery:
    - A stable prediction contract makes it easy to wire outputs into reports, APIs, or downstream decision systems.
    """
    config = load_config()
    problem_type = config["model"]["problem_type"].strip().lower()

    logger.info("Running inference using model.predict")

    if not isinstance(X_infer, pd.DataFrame):
        raise TypeError("Inference failed: X_infer must be a pandas DataFrame.")

    if not hasattr(model, "predict"):
        raise TypeError("Inference failed: model does not implement .predict(). Check your saved artifact.")

    preds = model.predict(X_infer)
    df_pred = pd.DataFrame({"prediction": preds}, index=X_infer.index)

    # --------------------------------------------------------
    # START STUDENT CODE
    # --------------------------------------------------------
    # Student logic: probabilities + label mapping (kept), but do NOT change output schema unless the team agrees.

    if problem_type == "classification":
        # Probability extraction (optional + safe)
        if hasattr(model, "predict_proba"):
            proba = model.predict_proba(X_infer)
            # Defensive shape check
            if getattr(proba, "ndim", 0) == 2 and proba.shape[1] >= 2:
                churn_probability = proba[:, 1]
                logger.info(
                    "churn_probability computed (min=%.4f, max=%.4f)",
                    churn_probability.min(),
                    churn_probability.max(),
                )

        # Label mapping (optional): keep as local mapping unless downstream expects strings
        label_map = {0: "No Churn", 1: "Churn"}
        _example_labels = pd.Series(df_pred["prediction"]).map(label_map).head(3).tolist()
        logger.info("Example mapped labels (first 3): %s", _example_labels)
    # --------------------------------------------------------
    # END STUDENT CODE
    # --------------------------------------------------------

    output_path = get_project_root() / config["data"]["inference_output"]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df_pred.to_csv(output_path, index=True)
    logger.info("Predictions saved -> %s", output_path)

    return df_pred
