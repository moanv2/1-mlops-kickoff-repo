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

import logging
from pathlib import Path

import joblib
import pandas as pd

from src.utils import get_project_root, load_config

logger = logging.getLogger(__name__)


def load_model_from_wandb(artifact_name: str, alias: str = "prod") -> object:
    """Download the promoted W&B model artifact and return the fitted pipeline.

    Parameters
    ----------
    artifact_name : str
        Full artifact path, e.g. 'entity/project/artifact:alias'
        or just 'artifact:alias' if a wandb run is active.
    alias : str
        Artifact alias to fetch (default 'prod').
    """
    import wandb

    api = wandb.Api()
    config = load_config()
    wandb_cfg = config.get("wandb", {})
    project = wandb_cfg.get("project", "MLOPs-Group5-telecomChurn")
    name = wandb_cfg.get("model_artifact", "telecom-churn-model")

    full_path = f"{project}/{name}:{alias}"
    logger.info("Fetching W&B artifact: %s", full_path)
    artifact = api.artifact(full_path)
    artifact_dir = artifact.download()

    # Find the .pkl file inside the artifact
    pkl_files = list(Path(artifact_dir).glob("*.pkl"))
    if not pkl_files:
        raise FileNotFoundError(f"No .pkl file found in artifact '{full_path}'")

    model = joblib.load(pkl_files[0])
    logger.info("Loaded model from W&B artifact: %s", pkl_files[0].name)
    return model


def run_inference(model, X_infer: pd.DataFrame, save: bool = True) -> pd.DataFrame:
    """
    Inputs:
    - model: Fitted sklearn Pipeline.
    - X_infer: Features DataFrame for inference.
    - save: If True, persist predictions to CSV (default for pipeline; False for API).
    Outputs:
    - df_pred: DataFrame with exactly one column named 'prediction', preserving X_infer index.
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

    if problem_type == "classification":
        # Probability extraction (optional + safe)
        if hasattr(model, "predict_proba"):
            proba = model.predict_proba(X_infer)
            if getattr(proba, "ndim", 0) == 2 and proba.shape[1] >= 2:
                churn_probability = proba[:, 1]
                logger.info(
                    "churn_probability computed (min=%.4f, max=%.4f)",
                    churn_probability.min(),
                    churn_probability.max(),
                )

        # Label mapping (optional)
        label_map = {0: "No Churn", 1: "Churn"}
        _example_labels = pd.Series(df_pred["prediction"]).map(label_map).head(3).tolist()
        logger.info("Example mapped labels (first 3): %s", _example_labels)

    if save:
        output_path = get_project_root() / config["data"]["inference_output"]
        output_path.parent.mkdir(parents=True, exist_ok=True)
        df_pred.to_csv(output_path, index=True)
        logger.info("Predictions saved -> %s", output_path)

    return df_pred
