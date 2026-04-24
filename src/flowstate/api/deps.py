"""Typed per-app state container and FastAPI dependency accessors.

Kept separate from routes so Phase 3 can depend on a stable surface: routes
pull `AppState` from the request and read `state.batcher` / `state.runtime`
without importing lifespan internals.
"""

from __future__ import annotations

from dataclasses import dataclass

from fastapi import Request

from flowstate.inference.batcher import DynamicBatcher
from flowstate.inference.runtime import ModelRuntime
from flowstate.inference.tokenizer import CachedTokenizer


@dataclass
class AppState:
    runtime: ModelRuntime | None = None
    tokenizer: CachedTokenizer | None = None
    batcher: DynamicBatcher | None = None
    model_loaded: bool = False
    cache_connected: bool = False
    ready: bool = False


def get_state(request: Request) -> AppState:
    state: AppState = request.app.state.flowstate
    return state
