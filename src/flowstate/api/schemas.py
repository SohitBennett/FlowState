"""Pydantic v2 request/response schemas. Strict validation at the boundary."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field


# `model_*` field names trip Pydantic v2's "protected namespace" warning on
# older 2.x; opt out per-model since `model_version` is part of the public API.
_RESPONSE_CONFIG = ConfigDict(protected_namespaces=())


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
    model_config = _RESPONSE_CONFIG

    model_version: str
    cached: bool
    prediction: Prediction
    latency_ms: float


class BatchPredictResponse(BaseModel):
    model_config = _RESPONSE_CONFIG

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


class ErrorBody(BaseModel):
    code: str
    message: str


class ErrorResponse(BaseModel):
    error: ErrorBody


class AdminReloadResponse(BaseModel):
    model_config = _RESPONSE_CONFIG

    status: str
    model_version: str | None
    ready: bool
