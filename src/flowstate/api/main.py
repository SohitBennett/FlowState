"""FastAPI application factory.

Lifespan: load the ONNX model + warm up + start the batcher (Phase 2 hooks),
then later install the Redis cache (Phase 4) and tracing exporters
(Phase 5). Middleware execution order, outermost first:

    RequestContextMiddleware  → request_id + timing + metrics
    BodySizeLimitMiddleware   → 413 on oversized requests
    AuthMiddleware            → 401 on missing/invalid x-api-key
    RateLimitMiddleware       → 429 on per-key budget exhaustion
    routes
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.responses import ORJSONResponse

from flowstate import __version__
from flowstate.api.deps import AppState
from flowstate.api.errors import FlowStateError, flowstate_error_handler
from flowstate.api.inference_lifecycle import load_inference
from flowstate.api.middleware import (
    AuthMiddleware,
    BodySizeLimitMiddleware,
    RateLimitMiddleware,
    RequestContextMiddleware,
)
from flowstate.api.rate_limit import RateLimiter
from flowstate.api.routes import admin, health, predict
from flowstate.config import get_settings
from flowstate.logging import configure_logging, get_logger


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    settings = get_settings()
    configure_logging(settings.log_level)
    log = get_logger(__name__)
    log.info("flowstate_starting", env=settings.env, version=__version__)

    state: AppState = app.state.flowstate
    state.reload_lock = asyncio.Lock()

    try:
        await load_inference(state, settings)
    except Exception:  # noqa: BLE001
        log.exception("inference_load_failed")

    try:
        yield
    finally:
        log.info("flowstate_stopping")
        state.ready = False
        if state.batcher is not None:
            await state.batcher.stop()


def create_app() -> FastAPI:
    settings = get_settings()
    app = FastAPI(
        title="FlowState Inference API",
        version=__version__,
        default_response_class=ORJSONResponse,
        lifespan=lifespan,
    )

    # Seed per-app state eagerly so endpoints work even without lifespan
    # (bare TestClient instantiation, schemathesis ASGI calls, etc.).
    app.state.flowstate = AppState()

    limiter = RateLimiter(rate=settings.rate_limit_rate, burst=settings.rate_limit_burst)

    # add_middleware is LIFO: the LAST registered middleware runs FIRST
    # (outermost). Order below produces:
    #   RequestContext > BodySizeLimit > Auth > RateLimit > endpoint
    app.add_middleware(RateLimitMiddleware, limiter=limiter)
    app.add_middleware(AuthMiddleware)
    app.add_middleware(BodySizeLimitMiddleware, max_bytes=settings.max_request_bytes)
    app.add_middleware(RequestContextMiddleware)

    app.add_exception_handler(FlowStateError, flowstate_error_handler)  # type: ignore[arg-type]

    app.include_router(health.router)
    app.include_router(predict.router)
    app.include_router(admin.router)
    return app


app = create_app()
