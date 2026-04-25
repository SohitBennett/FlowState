"""Single + batch prediction routes.

Both routes funnel through the dynamic batcher, so a single client and a
many-client workload share the same throughput-amortizing path.
"""

from __future__ import annotations

import asyncio
import time

from fastapi import APIRouter, Depends

from flowstate.api.deps import AppState, get_state
from flowstate.api.errors import ModelNotReadyError
from flowstate.api.schemas import (
    BatchPredictRequest,
    BatchPredictResponse,
    ErrorResponse,
    PredictRequest,
    PredictResponse,
    Prediction,
)

router = APIRouter(prefix="/v1", tags=["predict"])

_ERROR_RESPONSES: dict[int | str, dict[str, object]] = {
    401: {"model": ErrorResponse, "description": "Missing or invalid API key"},
    422: {"model": ErrorResponse, "description": "Request validation failed"},
    429: {"model": ErrorResponse, "description": "Rate limit exceeded"},
    503: {"model": ErrorResponse, "description": "Model not loaded"},
}


def _ensure_ready(state: AppState) -> None:
    if not state.ready or state.batcher is None:
        raise ModelNotReadyError("model not loaded")


@router.post(
    "/predict",
    response_model=PredictResponse,
    responses=_ERROR_RESPONSES,
    summary="Classify a single text input.",
)
async def predict(
    req: PredictRequest, state: AppState = Depends(get_state)
) -> PredictResponse:
    _ensure_ready(state)
    assert state.batcher is not None  # narrowing for mypy

    start = time.perf_counter()
    result = await state.batcher.submit(req.text)
    elapsed_ms = (time.perf_counter() - start) * 1000
    return PredictResponse(
        model_version=state.model_version or "unknown",
        cached=False,  # Phase 4 wires the cache and populates this.
        prediction=Prediction(label=result.label, score=result.score),
        latency_ms=elapsed_ms,
    )


@router.post(
    "/predict/batch",
    response_model=BatchPredictResponse,
    responses=_ERROR_RESPONSES,
    summary="Classify multiple text inputs in a single request.",
)
async def predict_batch(
    req: BatchPredictRequest, state: AppState = Depends(get_state)
) -> BatchPredictResponse:
    _ensure_ready(state)
    assert state.batcher is not None  # narrowing for mypy

    start = time.perf_counter()
    results = await asyncio.gather(*(state.batcher.submit(t) for t in req.texts))
    elapsed_ms = (time.perf_counter() - start) * 1000
    return BatchPredictResponse(
        model_version=state.model_version or "unknown",
        predictions=[Prediction(label=r.label, score=r.score) for r in results],
        latency_ms=elapsed_ms,
    )
