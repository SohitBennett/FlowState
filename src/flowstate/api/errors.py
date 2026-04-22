"""Typed API errors with consistent JSON shape."""

from __future__ import annotations

from fastapi import Request, status
from fastapi.responses import ORJSONResponse


class FlowStateError(Exception):
    status_code: int = status.HTTP_500_INTERNAL_SERVER_ERROR
    code: str = "internal_error"

    def __init__(self, message: str) -> None:
        super().__init__(message)
        self.message = message


class ModelNotReadyError(FlowStateError):
    status_code = status.HTTP_503_SERVICE_UNAVAILABLE
    code = "model_not_ready"


class UnauthorizedError(FlowStateError):
    status_code = status.HTTP_401_UNAUTHORIZED
    code = "unauthorized"


class RateLimitedError(FlowStateError):
    status_code = status.HTTP_429_TOO_MANY_REQUESTS
    code = "rate_limited"


async def flowstate_error_handler(_: Request, exc: FlowStateError) -> ORJSONResponse:
    return ORJSONResponse(
        status_code=exc.status_code,
        content={"error": {"code": exc.code, "message": exc.message}},
    )
