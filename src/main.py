"""
Module: Main Pipeline
---------------------
Role: Orchestrate the entire flow (Load -> Validate -> Clean -> Train -> Evaluate -> Infer).
Usage: python src/main.py
"""

import yaml
from pathlib import Path
from src.load_data import load_data
from src.validate import validate_dataframe
from src.clean_data import clean_data
from src.feature_engineering import build_features, FeatureConfig
from src.train import train_model
from src.evaluate import evaluate_model
from src.infer import run_inference
from sklearn.preprocessing import FunctionTransformer

CONFIG_PATH = Path("config.yaml")

with open(CONFIG_PATH) as f:
    config = yaml.safe_load(f)

REQUIRED_COLUMNS = config["validate"]["required_columns"]
TARGET_COL = config["features"]["target_col"]
NUMERIC_COLS = config["features"]["numeric_cols"]
CATEGORICAL_COLS = config["features"]["categorical_cols"]
PROBLEM_TYPE = config["train"]["problem_type"]
MODEL_PATH = config["train"]["model_path"]

if __name__ == "__main__":
    # Step 1: Load data
    df = load_data(Path(config["data"]["raw"]))

    # Step 2: Validate (before cleaning — columns still have original casing)
    validate_dataframe(df, REQUIRED_COLUMNS)

    # Step 3: Clean
    df = clean_data(df)

    # Step 4: Feature engineering (columns are now lowercase)
    cfg = FeatureConfig(
        target_col=TARGET_COL,
        numeric_cols=tuple(NUMERIC_COLS),
        categorical_cols=tuple(CATEGORICAL_COLS),
    )
    df = build_features(df, cfg)

    # Step 5: Split features and target
    y = df[TARGET_COL]
    X = df.drop(columns=[TARGET_COL])

    # Step 6: Train (FunctionTransformer passes features through — already engineered above)
    preprocessor = FunctionTransformer()
    fitted_pipeline, X_test, y_test = train_model(
        X, y, preprocessor, PROBLEM_TYPE, MODEL_PATH,
        test_size=config["train"]["test_size"],
        random_state=config["train"]["seed"],
    )

    # Step 7: Evaluate
    metric = evaluate_model(fitted_pipeline, X_test, y_test, PROBLEM_TYPE)
    print(f"Pipeline complete. {PROBLEM_TYPE} metric: {metric:.4f}")

    # Step 8: Infer
    predictions = run_inference(fitted_pipeline, X_test)
    print(predictions.head())
