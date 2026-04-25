"""Typed per-app state container and FastAPI dependency accessors.

Routes pull `AppState` via `get_state` so they don't need to know how the
runtime was loaded — the lifespan and the `POST /admin/reload` route both
mutate this same instance through `inference_lifecycle`.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field

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
    model_version: str | None = None
    reload_lock: asyncio.Lock | None = field(default=None)


def get_state(request: Request) -> AppState:
    state: AppState = request.app.state.flowstate
    return state
