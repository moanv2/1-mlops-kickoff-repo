"""
Module: Main Pipeline
---------------------
Role: Orchestrate the entire flow (Load -> Validate -> Clean -> Train -> Evaluate -> Infer).
Usage: python src/main.py
"""

from src.load_data import load_data
from src.validate import validate_dataframe
from src.clean_data import clean_data
from src.feature_engineering import build_features, FeatureConfig
from src.train import train_model
from src.evaluate import evaluate_model
from src.infer import run_inference
from sklearn.preprocessing import FunctionTransformer

REQUIRED_COLUMNS = [
    "AccountWeeks", "DataUsage", "CustServCalls",
    "DayMins", "DayCalls", "MonthlyCharge",
    "OverageFee", "RoamMins", "Churn",
    "ContractRenewal", "DataPlan",
]

TARGET_COL = "churn"       # lowercase after clean_data standardizes column names
PROBLEM_TYPE = "classification"
MODEL_PATH = "models/model.pkl"

if __name__ == "__main__":
    # Step 1: Load data
    df = load_data()

    # Step 2: Validate (before cleaning — columns still have original casing)
    validate_dataframe(df, REQUIRED_COLUMNS)

    # Step 3: Clean
    df = clean_data(df)

    # Step 4: Feature engineering (columns are now lowercase)
    cfg = FeatureConfig(
        target_col=TARGET_COL,
        numeric_cols=(
            "accountweeks", "datausage", "custservcalls",
            "daymins", "daycalls", "monthlycharge",
            "overagefee", "roammins",
        ),
        categorical_cols=("contractrenewal", "dataplan"),
    )
    df = build_features(df, cfg)

    # Step 5: Split features and target
    y = df[TARGET_COL]
    X = df.drop(columns=[TARGET_COL])

    # Step 6: Train (FunctionTransformer passes features through — already engineered above)
    preprocessor = FunctionTransformer()
    fitted_pipeline, X_test, y_test = train_model(
        X, y, preprocessor, PROBLEM_TYPE, MODEL_PATH
    )

    # Step 7: Evaluate
    metric = evaluate_model(fitted_pipeline, X_test, y_test, PROBLEM_TYPE)
    print(f"Pipeline complete. {PROBLEM_TYPE} metric: {metric:.4f}")

    # Step 8: Infer
    predictions = run_inference(fitted_pipeline, X_test)
    print(predictions.head())
