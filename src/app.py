"""
FastAPI prediction service.

Endpoints:
  GET  /health   -> liveness check + which model is loaded
  POST /predict  -> send the 8 feature values as JSON, get a diabetes prediction

Run locally:  uvicorn src.app:app --reload
Then open the interactive docs at http://127.0.0.1:8000/docs
"""
import os
import sys
from contextlib import asynccontextmanager

from fastapi import FastAPI
from pydantic import BaseModel, Field

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from predict import predict, load_model, load_metadata  # noqa: E402


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load the model once at startup so the first request isn't slow (and so
    the container fails fast if the model file is missing)."""
    load_model()
    yield


app = FastAPI(
    title="Diabetes Prediction API",
    description="Predicts diabetes onset from the Pima Indians feature set.",
    version="1.0.0",
    lifespan=lifespan,
)


class DiabetesFeatures(BaseModel):
    """Request body. Field names match the dataset columns exactly."""

    Pregnancies: float = Field(..., examples=[6])
    Glucose: float = Field(..., examples=[148])
    BloodPressure: float = Field(..., examples=[72])
    SkinThickness: float = Field(..., examples=[35])
    Insulin: float = Field(..., examples=[0])
    BMI: float = Field(..., examples=[33.6])
    DiabetesPedigreeFunction: float = Field(..., examples=[0.627])
    Age: float = Field(..., examples=[50])


@app.get("/health")
def health():
    """Simple liveness probe used by tests, Docker, and CI."""
    meta = load_metadata()
    return {
        "status": "ok",
        "model_loaded": True,
        "model_name": meta.get("model_name", "unknown"),
    }


@app.post("/predict")
def make_prediction(features: DiabetesFeatures):
    """Return {prediction, label, probability} for one patient."""
    return predict(features.model_dump())
