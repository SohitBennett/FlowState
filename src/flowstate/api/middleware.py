"""HTTP middleware: request id + timing + metrics, body-size guard, auth, rate limit.

Custom middleware sits OUTSIDE Starlette's ExceptionMiddleware, so any
exception raised here would skip our `add_exception_handler` registry and
become a bare 500. Each middleware therefore returns an `ORJSONResponse`
directly when rejecting a request, rather than raising a typed error.
"""

from __future__ import annotations

import time
import uuid
from collections.abc import Awaitable, Callable

import structlog
from fastapi.responses import ORJSONResponse
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response

from flowstate.api.rate_limit import RateLimiter
from flowstate.config import get_settings
from flowstate.monitoring.metrics import REQUEST_COUNT, REQUEST_LATENCY

log = structlog.get_logger(__name__)

PUBLIC_PATHS: frozenset[str] = frozenset(
    {"/healthz", "/readyz", "/metrics", "/docs", "/openapi.json", "/redoc"}
)


def _error_response(status: int, code: str, message: str) -> ORJSONResponse:
    return ORJSONResponse(
        status_code=status,
        content={"error": {"code": code, "message": message}},
    )


class RequestContextMiddleware(BaseHTTPMiddleware):
    """Outermost middleware: assigns a request_id, times the request, records metrics."""

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

        route_obj = request.scope.get("route")
        route = route_obj.path if route_obj is not None else request.url.path
        REQUEST_COUNT.labels(request.method, route, str(response.status_code)).inc()
        REQUEST_LATENCY.labels(route).observe(elapsed)

        response.headers["x-request-id"] = request_id
        response.headers["x-response-time-ms"] = f"{elapsed * 1000:.2f}"
        log.info("request_complete", status=response.status_code, latency_ms=elapsed * 1000)
        return response


class BodySizeLimitMiddleware(BaseHTTPMiddleware):
    """Reject requests whose declared Content-Length exceeds `max_bytes`.

    A defense-in-depth layer; the production deployment is also expected to
    cap body size at the proxy (NGINX/Traefik). Requests without a
    Content-Length header (chunked uploads) bypass this check.
    """

    def __init__(self, app: object, max_bytes: int) -> None:
        super().__init__(app)  # type: ignore[arg-type]
        self._max_bytes = max_bytes

    async def dispatch(
        self, request: Request, call_next: Callable[[Request], Awaitable[Response]]
    ) -> Response:
        cl = request.headers.get("content-length")
        if cl is not None:
            try:
                size = int(cl)
            except ValueError:
                size = 0
            if size > self._max_bytes:
                return _error_response(
                    413,
                    "request_too_large",
                    f"request body exceeds {self._max_bytes} bytes",
                )
        return await call_next(request)


class AuthMiddleware(BaseHTTPMiddleware):
    """API-key auth via x-api-key header. Public paths bypass."""

    async def dispatch(
        self, request: Request, call_next: Callable[[Request], Awaitable[Response]]
    ) -> Response:
        if request.url.path in PUBLIC_PATHS:
            return await call_next(request)
        provided = request.headers.get("x-api-key")
        if provided != get_settings().api_key:
            return _error_response(401, "unauthorized", "invalid or missing api key")
        return await call_next(request)


class RateLimitMiddleware(BaseHTTPMiddleware):
    """Per-API-key token-bucket rate limit. Public paths bypass."""

    def __init__(self, app: object, limiter: RateLimiter) -> None:
        super().__init__(app)  # type: ignore[arg-type]
        self._limiter = limiter

    async def dispatch(
        self, request: Request, call_next: Callable[[Request], Awaitable[Response]]
    ) -> Response:
        if request.url.path in PUBLIC_PATHS:
            return await call_next(request)
        api_key = request.headers.get("x-api-key", "")
        if not api_key:
            # AuthMiddleware will reject this request; pass through so the
            # 401 response is uniform.
            return await call_next(request)
        if not await self._limiter.check(api_key):
            return _error_response(429, "rate_limited", "rate limit exceeded")
        return await call_next(request)
