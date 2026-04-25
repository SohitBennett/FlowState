"""Authenticated admin operations: hot-swap the model from disk."""

from __future__ import annotations

from fastapi import APIRouter, Depends

from flowstate.api.deps import AppState, get_state
from flowstate.api.inference_lifecycle import reload_inference
from flowstate.api.schemas import AdminReloadResponse, ErrorResponse
from flowstate.config import Settings, get_settings

router = APIRouter(prefix="/admin", tags=["admin"])


@router.post(
    "/reload",
    response_model=AdminReloadResponse,
    responses={
        401: {"model": ErrorResponse, "description": "Missing or invalid API key"},
        429: {"model": ErrorResponse, "description": "Rate limit exceeded"},
    },
    summary="Reload the model from disk (hot swap).",
)
async def reload_model(
    state: AppState = Depends(get_state),
    settings: Settings = Depends(get_settings),
) -> AdminReloadResponse:
    await reload_inference(state, settings)
    return AdminReloadResponse(
        status="reloaded",
        model_version=state.model_version,
        ready=state.ready,
    )
