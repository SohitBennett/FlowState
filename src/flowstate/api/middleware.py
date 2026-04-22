"""HTTP middleware: request id, timing, auth, metrics."""

from __future__ import annotations

import time
import uuid
from collections.abc import Awaitable, Callable

import structlog
from fastapi import Request
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import Response

from flowstate.api.errors import UnauthorizedError
from flowstate.config import get_settings
from flowstate.monitoring.metrics import REQUEST_COUNT, REQUEST_LATENCY

log = structlog.get_logger(__name__)

PUBLIC_PATHS = {"/healthz", "/readyz", "/metrics", "/docs", "/openapi.json", "/redoc"}


class RequestContextMiddleware(BaseHTTPMiddleware):
    async def dispatch(
        self, request: Request, call_next: Callable[[Request], Awaitable[Response]]
    ) -> Response:
        request_id = request.headers.get("x-request-id") or uuid.uuid4().hex
        structlog.contextvars.clear_contextvars()
        structlog.contextvars.bind_contextvars(
            request_id=request_id,
            method=request.method,
            path=request.url.path,
        )

        start = time.perf_counter()
        try:
            response = await call_next(request)
        except Exception:
            log.exception("unhandled_error")
            raise
        elapsed = time.perf_counter() - start

        route = request.scope.get("route").path if request.scope.get("route") else request.url.path
        REQUEST_COUNT.labels(request.method, route, str(response.status_code)).inc()
        REQUEST_LATENCY.labels(route).observe(elapsed)

        response.headers["x-request-id"] = request_id
        response.headers["x-response-time-ms"] = f"{elapsed * 1000:.2f}"
        log.info("request_complete", status=response.status_code, latency_ms=elapsed * 1000)
        return response


class AuthMiddleware(BaseHTTPMiddleware):
    """Simple API-key auth. JWT support slots in here in Phase 3."""

    async def dispatch(
        self, request: Request, call_next: Callable[[Request], Awaitable[Response]]
    ) -> Response:
        if request.url.path in PUBLIC_PATHS:
            return await call_next(request)

        provided = request.headers.get("x-api-key")
        if provided != get_settings().api_key:
            raise UnauthorizedError("invalid or missing api key")
        return await call_next(request)
