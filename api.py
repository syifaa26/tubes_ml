from __future__ import annotations

from functools import lru_cache
from typing import Annotated, Any

from fastapi import FastAPI, Query
from pydantic import BaseModel, Field

from src.config import DEFAULT_MODEL_NAME, DEFAULT_THRESHOLD, FEATURE_NAMES, MODEL_OPTIONS
from src.model_utils import load_model, load_scaler, predict_transactions


app = FastAPI(
    title="Fraud Detection API",
    description="REST API untuk prediksi fraud transaksi.",
    version="1.0.0",
)


class Transaction(BaseModel):
    distance_from_home: float = Field(..., ge=0)
    distance_from_last_transaction: float = Field(..., ge=0)
    ratio_to_median_purchase_price: float = Field(..., ge=0)
    repeat_retailer: int = Field(..., ge=0, le=1)
    used_chip: int = Field(..., ge=0, le=1)
    used_pin_number: int = Field(..., ge=0, le=1)
    online_order: int = Field(..., ge=0, le=1)


ModelNameQuery = Annotated[str, Query(enum=list(MODEL_OPTIONS.keys()))]
ThresholdQuery = Annotated[float, Query(ge=0.0, le=1.0)]


@lru_cache(maxsize=4)
def get_model(model_name: str):
    return load_model(model_name)


@lru_cache(maxsize=1)
def get_scaler():
    return load_scaler()


def to_dict(model: BaseModel) -> dict[str, Any]:
    if hasattr(model, "model_dump"):
        return model.model_dump()
    return model.dict()


@app.get("/")
def root() -> dict[str, Any]:
    return {
        "message": "Fraud Detection API aktif",
        "docs": "/docs",
        "models": list(MODEL_OPTIONS.keys()),
        "features": FEATURE_NAMES,
    }


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/predict")
def predict(
    transaction: Transaction,
    model_name: ModelNameQuery = DEFAULT_MODEL_NAME,
    threshold: ThresholdQuery = DEFAULT_THRESHOLD,
) -> dict[str, Any]:
    model = get_model(model_name)
    scaler = get_scaler()
    result = predict_transactions(model, to_dict(transaction), threshold=threshold, scaler=scaler).iloc[0]
    return {
        "model": model_name,
        "threshold": threshold,
        "prediction": int(result["prediction"]),
        "prediction_label": str(result["prediction_label"]),
        "fraud_probability": float(result["fraud_probability"]),
        "fraud_risk_percent": float(result["fraud_risk_percent"]),
    }


@app.post("/predict-batch")
def predict_batch(
    transactions: list[Transaction],
    model_name: ModelNameQuery = DEFAULT_MODEL_NAME,
    threshold: ThresholdQuery = DEFAULT_THRESHOLD,
) -> dict[str, Any]:
    model = get_model(model_name)
    scaler = get_scaler()
    records = [to_dict(transaction) for transaction in transactions]
    result = predict_transactions(model, records, threshold=threshold, scaler=scaler)
    return {
        "model": model_name,
        "threshold": threshold,
        "total": int(len(result)),
        "fraud": int((result["prediction"] == 1).sum()),
        "non_fraud": int((result["prediction"] == 0).sum()),
        "items": result.to_dict(orient="records"),
    }
