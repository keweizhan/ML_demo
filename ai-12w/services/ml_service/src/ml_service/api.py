from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import joblib
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from .config import settings
from .logging_config import setup_logging

setup_logging(settings.log_level)
log = logging.getLogger("ml_service")

app = FastAPI(title="ml_service")

# Lazy-loaded model cache
_model: Any | None = None


class PredictRequest(BaseModel):
    # Single sample: [f1, f2, ...]
    # Batch: [[...], [...]]
    features: list[float] | list[list[float]] = Field(
        ..., description="Single sample or batch of samples"
    )


class PredictResponse(BaseModel):
    prediction: int | list[int]
    probability: float | list[float] | None = None


def _load_model() -> Any:
    global _model
    if _model is not None:
        return _model

    model_path = Path(settings.model_path)
    if not model_path.exists():
        raise FileNotFoundError(
            f"Model file not found at {model_path}. Run: uv run python -m ml_service.train"
        )

    _model = joblib.load(model_path)
    log.info("Loaded model from %s", model_path)
    return _model


@app.get("/health")
def health() -> dict:
    return {
        "status": "ok",
        "env": settings.env,
        "model_path": settings.model_path,
        "model_loaded": _model is not None,
    }


@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest) -> PredictResponse:
    try:
        model = _load_model()
    except FileNotFoundError as e:
        raise HTTPException(status_code=409, detail=str(e)) from e
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to load model: {e}") from e

    features = req.features
    is_batch = isinstance(features, list) and len(features) > 0 and isinstance(features[0], list)
    X = features if is_batch else [features]  # type: ignore[list-item]

    try:
        preds = model.predict(X)
        probs: list[float] | None = None

        if hasattr(model, "predict_proba"):
            p = model.predict_proba(X)
            probs = [float(row[1]) for row in p]

        if is_batch:
            return PredictResponse(
                prediction=[int(x) for x in preds],
                probability=probs,
            )

        return PredictResponse(
            prediction=int(preds[0]),
            probability=(probs[0] if probs is not None else None),
        )
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Bad input or prediction failed: {e}") from e
