"""
Module: Model Training
----------------------
Role: Split data, train model, and save the artifact.
Input: pandas.DataFrame (Processed).
Output: Serialized model file (e.g., .pkl) in `models/`.
"""

"""
Educational Goal:

Why this module exists in an MLOps system: Training is the core step where
a model learns patterns from data. In a production MLOps system, this step
must be reproducible, isolated from data leakage, and produce a
deployment-ready artifact that bundles both preprocessing and the model.

Responsibility (separation of concerns): This module is responsible for
splitting data into train/test sets, fitting a Scikit-Learn Pipeline on the
training split, saving the fitted pipeline as a .pkl artifact, and returning
the test split for downstream evaluation.

Pipeline contract (inputs and outputs):
  Inputs:  X (pd.DataFrame)        — full feature matrix (all rows)
           y (pd.Series)           — full target vector (all rows)
           preprocessor            — a fitted-or-unfitted ColumnTransformer
                                     (will be fit inside the Pipeline)
           problem_type (str)      — "regression" or "classification"
           model_path (str)        — destination path for the .pkl artifact
           test_size (float)       — fraction of data held out for testing
           random_state (int)      — seed for reproducible splits
  Outputs: fitted_pipeline         — fitted sklearn.pipeline.Pipeline
           X_test (pd.DataFrame)   — held-out feature matrix
           y_test (pd.Series)      — held-out target vector

TODO: Replace print statements with standard library logging in a later session
TODO: Any temporary or hardcoded variable or parameter will be imported from config.yml in a later session
"""

import os
import pandas as pd
import joblib
from sklearn.pipeline import Pipeline
from sklearn.linear_model import Ridge, LogisticRegression
from sklearn.model_selection import cross_val_score, train_test_split


def train_model(
    X: pd.DataFrame,
    y: pd.Series,
    preprocessor,
    problem_type: str,
    model_path: str = "models/model.pkl",
    test_size: float = 0.2,
    random_state: int = 42,
) -> tuple:
    """
    Inputs:

        X            (pd.DataFrame) : Full feature matrix (all rows).
                                      Must not be empty.
        y            (pd.Series)    : Full target vector aligned row-for-row
                                      with X. Must have the same length.
        preprocessor                : A Scikit-Learn ColumnTransformer (or any
                                      transformer) that handles numeric scaling
                                      and categorical encoding. It will be fit
                                      inside the Pipeline to prevent leakage.
        problem_type (str)          : Task type — must be "regression" or
                                      "classification". Controls which
                                      estimator is added to the Pipeline.
        model_path   (str)          : File path where the fitted pipeline will
                                      be saved as a .pkl artifact. The parent
                                      directory is created if it does not exist.
        test_size    (float)        : Proportion of data reserved for testing
                                      (default 0.2 = 20%).
        random_state (int)          : Random seed for reproducible splits
                                      (default 42).

    Outputs:

        fitted_pipeline (Pipeline)  : A Scikit-Learn Pipeline with two named
                                      steps — "preprocess" and "model" — fit
                                      on the training split only.
        X_test (pd.DataFrame)       : Held-out feature matrix for evaluation.
        y_test (pd.Series)          : Held-out target vector for evaluation.

    Why this contract matters for reliable ML delivery:

        Bundling preprocessing and the model into a single Pipeline object
        guarantees that every prediction — whether during evaluation, staging,
        or production — passes through exactly the same transformation logic.
        This eliminates a whole class of training/serving skew bugs. Enforcing
        that .fit() is called only on the training split maintains leakage
        boundaries, making evaluation metrics trustworthy. Saving the artifact
        with joblib ensures the pipeline can be loaded and served without
        re-training.
    """
    print(f"[train_model] Starting model training for problem type: '{problem_type}'")  # TODO: replace with logging later

    # ------------------------------------------------------------------
    # Fail-fast guardrails — catch bad inputs before sklearn sees them
    # ------------------------------------------------------------------
    if X.empty:
        raise ValueError(
            "[train_model] X is empty. Cannot train on zero samples. "
            "Check that your data loading and cleaning steps produced a "
            "non-empty DataFrame."
        )

    if len(X) != len(y):
        raise ValueError(
            f"[train_model] Shape mismatch: X has {len(X)} rows "
            f"but y has {len(y)} entries. They must be aligned "
            "row-for-row."
        )

    # ------------------------------------------------------------------
    # Train / test split
    # ------------------------------------------------------------------
    print(f"[train_model] Splitting data: test_size={test_size}, random_state={random_state}")  # TODO: replace with logging later

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

    print(f"[train_model] Train size: {len(X_train)} rows | Test size: {len(X_test)} rows")  # TODO: replace with logging later

    # ------------------------------------------------------------------
    # Model selection based on problem type
    # ------------------------------------------------------------------
    print(f"[train_model] Selecting estimator for problem type: '{problem_type}'")  # TODO: replace with logging later

    problem_type_normalized = problem_type.strip().lower()

    if problem_type_normalized == "regression":
        estimator = Ridge()  # TODO: alpha will be imported from config.yml later

    elif problem_type_normalized == "classification":
        # solver='liblinear' is deterministic and well-suited for small-to-medium
        # datasets; random_state=42 ensures reproducible results across runs.
        estimator = LogisticRegression(
            solver="liblinear",
            max_iter=500,
            random_state=42,  # TODO: random_state will be imported from config.yml later
        )

    else:
        raise ValueError(
            f"[train_model] Unsupported problem_type: '{problem_type}'. "
            "Expected 'regression' or 'classification'."
        )

    # ------------------------------------------------------------------
    # Build the unified Pipeline artifact
    # ------------------------------------------------------------------
    print("[train_model] Building sklearn Pipeline with steps: ['preprocess', 'model']")  # TODO: replace with logging later

    pipeline = Pipeline(
        steps=[
            ("preprocess", preprocessor),
            ("model", estimator),
        ]
    )

    # --------------------------------------------------------
    # START STUDENT CODE
    # --------------------------------------------------------
    # TODO_STUDENT: Paste your notebook logic here to replace or extend the baseline
    # Why: The right model and its hyperparameters depend heavily on the
    # dataset size, feature types, class imbalance, and business requirements.
    # The baseline (Ridge / LogisticRegression) is intentionally simple so you
    # can swap it out once you understand the data.
    #
    # Examples:
    # 1. Replace Ridge with RandomForestRegressor(n_estimators=100, random_state=42)
    #    when the target relationship is non-linear.
    # 2. Replace LogisticRegression with GradientBoostingClassifier(random_state=42)
    #    when you have imbalanced classes or complex decision boundaries.
    #
    # Optional forcing function (leave commented):
    # raise NotImplementedError("Student: You must implement this logic to proceed!")

    # --- Baseline implementation (dev-ready) ---
    # Report which estimator is active so the dev can confirm model selection.
    print(f"[train_model] Active estimator: {estimator.__class__.__name__}")  # TODO: replace with logging later
    print(f"[train_model] Training features: {X_train.shape[1]} columns, {len(X_train)} rows")  # TODO: replace with logging later

    # Cross-validation sanity check on the training split only (no leakage).
    # This gives an early signal on whether the pipeline generalises within
    # the training data before committing to a full fit.
    scoring = "r2" if problem_type_normalized == "regression" else "accuracy"
    cv_scores = cross_val_score(pipeline, X_train, y_train, cv=5, scoring=scoring)
    print(  # TODO: replace with logging later
        f"[train_model] 5-fold CV {scoring} on training split: "
        f"mean={cv_scores.mean():.4f}, std={cv_scores.std():.4f}"
    )
    # --------------------------------------------------------
    # END STUDENT CODE
    # --------------------------------------------------------

    # ------------------------------------------------------------------
    # Fit the Pipeline — ONLY on the training split (leakage boundary)
    # ------------------------------------------------------------------
    print(f"[train_model] Fitting pipeline on {len(X_train)} training samples...")  # TODO: replace with logging later

    fitted_pipeline = pipeline.fit(X_train, y_train)

    # ------------------------------------------------------------------
    # Save the artifact
    # ------------------------------------------------------------------
    os.makedirs(os.path.dirname(model_path) or ".", exist_ok=True)
    joblib.dump(fitted_pipeline, model_path)
    print(f"[train_model] Artifact saved to '{model_path}'")  # TODO: replace with logging later

    print("[train_model] Training complete. Returning fitted Pipeline and test split.")  # TODO: replace with logging later

    return fitted_pipeline, X_test, y_test
