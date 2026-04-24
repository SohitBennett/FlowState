"""Liveness, readiness, and metrics endpoints."""

from __future__ import annotations

from fastapi import APIRouter, Depends, Response

from flowstate import __version__
from flowstate.api.deps import AppState, get_state
from flowstate.api.schemas import HealthResponse, ReadyResponse
from flowstate.monitoring.metrics import render_metrics

router = APIRouter(tags=["health"])


@router.get("/healthz", response_model=HealthResponse)
async def healthz() -> HealthResponse:
    """Liveness — process is up. Cheap; never touches dependencies."""
    return HealthResponse(status="ok", version=__version__)


@router.get("/readyz", response_model=ReadyResponse)
async def readyz(state: AppState = Depends(get_state)) -> ReadyResponse:
    """Readiness — model loaded and warmed; cache wiring lands in Phase 4."""
    return ReadyResponse(
        status="ok" if state.ready else "starting",
        model_loaded=state.model_loaded,
        cache_connected=state.cache_connected,
    )


@router.get("/metrics")
async def metrics() -> Response:
    body, content_type = render_metrics()
    return Response(content=body, media_type=content_type)
