"""
Educational Goal:
- Why this module exists in an MLOps system: Provide a deployment-like prediction interface separate from training and evaluation.
- Responsibility (separation of concerns): Run model.predict on new data and return a standardized predictions DataFrame.
- Pipeline contract (inputs and outputs): Input fitted model + features; output DataFrame with one column "prediction".

TODO: Replace print statements with standard library logging in a later session
TODO: Any temporary or hardcoded variable or parameter will be imported from config.yml in a later session
"""

import pandas as pd


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
    print("[infer.run_inference] Running inference using model.predict")  # TODO: replace with logging later

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

    # Probability extraction (optional + safe)
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(X_infer)
        # Defensive shape check
        if getattr(proba, "ndim", 0) == 2 and proba.shape[1] >= 2:
            churn_probability = proba[:, 1]
            print(
                f"[infer.run_inference] churn_probability computed (min={churn_probability.min():.4f}, "
                f"max={churn_probability.max():.4f})"
            )  # TODO: replace with logging later

    # Label mapping (optional): keep as local mapping unless downstream expects strings
    label_map = {0: "No Churn", 1: "Churn"}
    _example_labels = pd.Series(df_pred["prediction"]).map(label_map).head(3).tolist()
    print(f"[infer.run_inference] Example mapped labels (first 3): {_example_labels}")  # TODO: replace with logging later
    # --------------------------------------------------------
    # END STUDENT CODE
    # --------------------------------------------------------

    return df_pred