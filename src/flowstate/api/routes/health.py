"""Liveness, readiness, and metrics endpoints."""

from __future__ import annotations

from fastapi import APIRouter, Response

from flowstate import __version__
from flowstate.api.schemas import HealthResponse, ReadyResponse
from flowstate.monitoring.metrics import render_metrics

router = APIRouter(tags=["health"])


@router.get("/healthz", response_model=HealthResponse)
async def healthz() -> HealthResponse:
    """Liveness — process is up. Cheap; never touches dependencies."""
    return HealthResponse(status="ok", version=__version__)


@router.get("/readyz", response_model=ReadyResponse)
async def readyz() -> ReadyResponse:
    """Readiness — model loaded + cache reachable. Wired up in Phase 2/4."""
    # Phase 0 stub: real checks land with the inference runtime + cache.
    return ReadyResponse(status="ok", model_loaded=False, cache_connected=False)


@router.get("/metrics")
async def metrics() -> Response:
    body, content_type = render_metrics()
    return Response(content=body, media_type=content_type)
