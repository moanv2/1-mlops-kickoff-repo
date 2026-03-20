"""
Module: API Serving
-------------------
Role: Serve predictions via FastAPI. Contains NO machine learning logic.
      Delegates to clean_data, feature_engineering, and infer modules.
Usage: uvicorn src.api:app --host 0.0.0.0 --port 8000
"""

from contextlib import asynccontextmanager
from typing import List

import dotenv
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from src.feature_engineering import FeatureConfig, build_features
from src.infer import load_model_from_wandb, run_inference
from src.logger import get_logger, setup_logging
from src.utils import load_config

dotenv.load_dotenv()

logger = get_logger(__name__)

# ── Pydantic request/response contracts ──────────────────────


class CustomerData(BaseModel):
    """Schema for a single customer record (matches feature columns)."""
    accountweeks: float = Field(..., ge=0, description="Weeks as customer")
    datausage: float = Field(..., ge=0, description="Monthly data usage (GB)")
    custservcalls: float = Field(..., ge=0, description="Customer service calls")
    daymins: float = Field(..., ge=0, description="Daytime minutes used")
    daycalls: float = Field(..., ge=0, description="Number of daytime calls")
    monthlycharge: float = Field(..., ge=0, description="Monthly charge ($)")
    overagefee: float = Field(..., ge=0, description="Overage fee ($)")
    roammins: float = Field(..., ge=0, description="Roaming minutes used")
    contractrenewal: int = Field(..., ge=0, le=1, description="Contract renewed (0/1)")
    dataplan: int = Field(..., ge=0, le=1, description="Has data plan (0/1)")


class PredictionRequest(BaseModel):
    """Batch prediction request — one or more customer records."""
    customers: List[CustomerData]


class PredictionResult(BaseModel):
    """Single prediction output."""
    prediction: int
    label: str


class PredictionResponse(BaseModel):
    """Batch prediction response."""
    predictions: List[PredictionResult]


class HealthResponse(BaseModel):
    """Health check response."""
    status: str
    model_loaded: bool


# ── Application state ────────────────────────────────────────

_model = None
_config = None


def _get_config():
    """Return cached config."""
    global _config
    if _config is None:
        _config = load_config()
    return _config


def _get_model():
    """Return the cached model, loading from W&B on first call."""
    global _model
    if _model is None:
        logger.info("Loading model from W&B artifact (alias='prod')...")
        _model = load_model_from_wandb(artifact_name="telecom-churn-model", alias="prod")
        logger.info("Model loaded successfully")
    return _model


def _prepare_features(df: pd.DataFrame) -> pd.DataFrame:
    """Apply the same clean + feature engineering as the training pipeline.

    Aligns output columns with what the model expects (handles missing
    one-hot columns that occur when a batch doesn't contain every category).
    No ML logic here — delegates to existing pipeline modules.
    """
    config = _get_config()
    # Pydantic already validated and typed the input; columns are already
    # lowercase from the schema. Only feature engineering is needed.
    cfg = FeatureConfig(
        numeric_cols=tuple(config["features"]["numeric"]),
        categorical_cols=tuple(config["features"]["categorical"]),
    )
    df = build_features(df, cfg)

    # Align columns with the model's expected features
    model = _get_model()
    if hasattr(model, "feature_names_in_"):
        expected = list(model.feature_names_in_)
        # Add missing columns as 0, drop extra columns
        for col in expected:
            if col not in df.columns:
                df[col] = 0
        df = df[expected]

    return df


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load model on startup."""
    config = _get_config()
    log_cfg = config.get("logging", {})
    setup_logging(
        log_file=log_cfg.get("log_file", "logs/pipeline.log"),
        level=log_cfg.get("level", "INFO"),
    )
    logger.info("Starting API server")
    _get_model()
    yield
    logger.info("Shutting down API server")


# ── FastAPI app ──────────────────────────────────────────────

app = FastAPI(
    title="Telecom Churn Prediction API",
    description="Predict customer churn using a promoted W&B model artifact.",
    version="1.0.0",
    lifespan=lifespan,
)


@app.get("/health", response_model=HealthResponse)
def health():
    """Health check endpoint."""
    return HealthResponse(
        status="healthy",
        model_loaded=_model is not None,
    )


@app.post("/predict", response_model=PredictionResponse)
def predict(request: PredictionRequest):
    """Return churn predictions for one or more customers.

    No ML logic here — delegates to existing pipeline modules.
    """
    model = _get_model()

    # Convert Pydantic models to DataFrame
    rows = [c.model_dump() for c in request.customers]
    df = pd.DataFrame(rows)

    logger.info("Received prediction request: %d customer(s)", len(df))

    try:
        # Apply same feature engineering as training pipeline
        df = _prepare_features(df)
        preds_df = run_inference(model, df, save=False)
    except Exception as e:
        logger.error("Inference failed: %s", e)
        raise HTTPException(status_code=422, detail=str(e))

    label_map = {0: "No Churn", 1: "Churn"}
    results = [
        PredictionResult(
            prediction=int(row["prediction"]),
            label=label_map.get(int(row["prediction"]), "Unknown"),
        )
        for _, row in preds_df.iterrows()
    ]

    return PredictionResponse(predictions=results)
