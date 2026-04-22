"""FastAPI application factory. Lifespan-managed resources land here in Phase 2+."""

from __future__ import annotations

from contextlib import asynccontextmanager
from collections.abc import AsyncIterator

from fastapi import FastAPI
from fastapi.responses import ORJSONResponse

from flowstate import __version__
from flowstate.api.errors import FlowStateError, flowstate_error_handler
from flowstate.api.middleware import AuthMiddleware, RequestContextMiddleware
from flowstate.api.routes import health
from flowstate.config import get_settings
from flowstate.logging import configure_logging, get_logger


@asynccontextmanager
async def lifespan(_: FastAPI) -> AsyncIterator[None]:
    settings = get_settings()
    configure_logging(settings.log_level)
    log = get_logger(__name__)
    log.info("flowstate_starting", env=settings.env, version=__version__)
    # Phase 2: load ONNX model + warmup here.
    # Phase 4: open Redis pool here.
    yield
    log.info("flowstate_stopping")


def create_app() -> FastAPI:
    app = FastAPI(
        title="FlowState Inference API",
        version=__version__,
        default_response_class=ORJSONResponse,
        lifespan=lifespan,
    )

    app.add_middleware(AuthMiddleware)
    app.add_middleware(RequestContextMiddleware)

    app.add_exception_handler(FlowStateError, flowstate_error_handler)  # type: ignore[arg-type]

    app.include_router(health.router)
    return app


app = create_app()
