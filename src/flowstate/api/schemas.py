"""Pydantic v2 request/response schemas. Strict validation at the boundary."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field


class PredictRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    text: str = Field(..., min_length=1, max_length=4096)


class BatchPredictRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    texts: list[str] = Field(..., min_length=1, max_length=128)


class Prediction(BaseModel):
    label: str
    score: float = Field(..., ge=0.0, le=1.0)


class PredictResponse(BaseModel):
    model_version: str
    cached: bool
    prediction: Prediction
    latency_ms: float


class BatchPredictResponse(BaseModel):
    model_version: str
    predictions: list[Prediction]
    latency_ms: float


class HealthResponse(BaseModel):
    status: str
    version: str


class ReadyResponse(BaseModel):
    status: str
    model_loaded: bool
    cache_connected: bool
