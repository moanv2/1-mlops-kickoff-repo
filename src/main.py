"""
Module: Main Pipeline
---------------------
Role: Orchestrate the entire flow (Load -> Validate -> Clean -> Train -> Evaluate -> Infer).
      W&B tracking is centralized here — no other module initializes wandb.
Usage: python -m src.main
"""

import os
from pathlib import Path

import dotenv
import wandb
import yaml
from sklearn.preprocessing import FunctionTransformer

from src.clean_data import clean_data
from src.evaluate import evaluate_model
from src.feature_engineering import FeatureConfig, build_features
from src.infer import run_inference
from src.load_data import load_data
from src.logger import get_logger, setup_logging
from src.train import train_model
from src.validate import validate_dataframe

# Load .env so WANDB_API_KEY is available
dotenv.load_dotenv()

logger = get_logger(__name__)

CONFIG_PATH = Path("config.yaml")

with open(CONFIG_PATH) as f:
    config = yaml.safe_load(f)


def run_pipeline(config: dict) -> float:
    """Run the full ML pipeline. Returns the evaluation metric."""
    # Initialize logging from config
    log_cfg = config.get("logging", {})
    setup_logging(
        log_file=log_cfg.get("log_file", "logs/pipeline.log"),
        level=log_cfg.get("level", "INFO"),
    )

    logger.info("Starting ML pipeline")

    # ── W&B initialization (centralized here) ────────────────
    wandb_cfg = config.get("wandb", {})
    run = wandb.init(
        project=wandb_cfg.get("project", "MLOPs-Group5-telecomChurn"),
        entity=wandb_cfg.get("entity"),
        job_type=wandb_cfg.get("job_type", "training-pipeline"),
        config={
            "problem_type": config["train"]["problem_type"],
            "test_size": config["train"]["test_size"],
            "val_size": config["train"].get("val_size", 0.2),
            "seed": config["train"]["seed"],
            "model_params": config["train"].get("model_params", {}),
            "features_numeric": config["features"]["numeric"],
            "features_categorical": config["features"]["categorical"],
            "target": config["target"]["column"],
        },
    )
    logger.info("W&B run initialized: %s", run.url)

    # Step 1: Load data
    df = load_data(Path(config["data"]["raw"]))
    wandb.log({"data/raw_rows": len(df), "data/raw_cols": len(df.columns)})

    # Step 2: Validate (before cleaning — columns still have original casing)
    validate_dataframe(df, config["validation"]["required_columns"])

    # Step 3: Clean
    df = clean_data(df, config)
    wandb.log({"data/clean_rows": len(df)})

    # Step 4: Feature engineering (columns are now lowercase)
    cfg = FeatureConfig(
        target_col=config["target"]["column"],
        numeric_cols=tuple(config["features"]["numeric"]),
        categorical_cols=tuple(config["features"]["categorical"]),
    )
    df = build_features(df, cfg)
    wandb.log({"data/feature_cols": len(df.columns)})

    # Step 5: Split features and target
    target_col = config["target"]["column"]
    y = df[target_col]
    X = df.drop(columns=[target_col])

    # Step 6: Train
    preprocessor = FunctionTransformer()
    fitted_pipeline, X_val, y_val, X_test, y_test = train_model(
        X, y, preprocessor,
        config["train"]["problem_type"],
        config["train"]["model_path"],
        test_size=config["train"]["test_size"],
        random_state=config["train"]["seed"],
    )
    wandb.log({
        "train/train_samples": len(X) - len(X_val) - len(X_test),
        "train/val_samples": len(X_val),
        "train/test_samples": len(X_test),
    })

    # Step 7: Evaluate
    metric = evaluate_model(fitted_pipeline, X_test, y_test, config["train"]["problem_type"])
    problem_type = config["train"]["problem_type"]
    metric_name = "f1_weighted" if problem_type == "classification" else "rmse"
    wandb.log({f"eval/{metric_name}": metric})
    logger.info("Pipeline complete. %s metric: %.4f", problem_type, metric)

    # Log evaluation plots to W&B
    figures_dir = Path(config.get("reports", {}).get("figures_dir", "reports/figures"))
    for plot_file in figures_dir.glob("*.png"):
        wandb.log({f"eval/{plot_file.stem}": wandb.Image(str(plot_file))})
        logger.info("Logged plot to W&B: %s", plot_file.name)

    # ── W&B model artifact (log + promote to 'prod') ────────
    model_path = config["train"]["model_path"]
    artifact_name = wandb_cfg.get("model_artifact", "telecom-churn-model")
    artifact = wandb.Artifact(
        name=artifact_name,
        type="model",
        description=f"Trained {problem_type} pipeline — {metric_name}={metric:.4f}",
        metadata={"metric_name": metric_name, "metric_value": metric},
    )
    artifact.add_file(model_path)
    run.log_artifact(artifact, aliases=["latest", "prod"])
    logger.info("Model artifact '%s' logged to W&B with alias 'prod'", artifact_name)

    # Log the pipeline log file to W&B for traceability
    log_file = log_cfg.get("log_file", "logs/pipeline.log")
    if Path(log_file).exists():
        log_artifact = wandb.Artifact(
            name="pipeline-logs",
            type="logs",
            description="Pipeline execution log",
        )
        log_artifact.add_file(log_file)
        run.log_artifact(log_artifact)
        logger.info("Pipeline log artifact uploaded to W&B")

    # Step 8: Infer
    predictions = run_inference(fitted_pipeline, X_test)
    logger.info("Inference sample:\n%s", predictions.head())

    wandb.finish()
    logger.info("W&B run finished")

    return metric


if __name__ == "__main__":
    run_pipeline(config)
