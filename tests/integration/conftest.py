"""Shared fixtures for API integration tests.

Each test gets a fresh app with `get_state` overridden to point at a fake
batcher. This lets us exercise routes, middleware, and validation without
loading a real ONNX model.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator
from typing import Any

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from flowstate.api.deps import AppState, get_state
from flowstate.api.main import create_app
from flowstate.config import get_settings
from flowstate.inference.postprocess import ClassificationResult


API_KEY = "test-key-for-integration-tests"


class FakeBatcher:
    """Deterministic batcher that always returns the first label with score 0.95."""

    def __init__(self, labels: tuple[str, ...] = ("World", "Sports", "Business", "Sci/Tech")) -> None:
        self.labels = labels
        self.calls = 0

    async def submit(self, text: str) -> ClassificationResult:
        self.calls += 1
        if not text:
            raise ValueError("empty text")
        return ClassificationResult(label=self.labels[0], score=0.95)

    async def stop(self, *_: Any, **__: Any) -> None:
        return None


class ExplodingBatcher:
    async def submit(self, _: str) -> ClassificationResult:
        raise RuntimeError("model exploded")

    async def stop(self, *_: Any, **__: Any) -> None:
        return None


@pytest.fixture(autouse=True)
def _isolate_settings(monkeypatch: pytest.MonkeyPatch) -> None:
    """Reset cached settings between tests with controlled env."""
    monkeypatch.setenv("FLOWSTATE_API_KEY", API_KEY)
    monkeypatch.setenv("FLOWSTATE_RATE_LIMIT_RATE", "1000")
    monkeypatch.setenv("FLOWSTATE_RATE_LIMIT_BURST", "1000")
    monkeypatch.setenv("FLOWSTATE_MAX_REQUEST_BYTES", "1048576")
    get_settings.cache_clear()
    yield
    get_settings.cache_clear()


def _make_state(batcher: object, model_version: str = "test:v1") -> AppState:
    return AppState(
        batcher=batcher,  # type: ignore[arg-type]
        model_loaded=True,
        ready=True,
        model_version=model_version,
    )


@pytest.fixture
def app_factory() -> Callable[..., FastAPI]:
    def _factory(
        batcher: object | None = None,
        model_version: str = "test:v1",
    ) -> FastAPI:
        app = create_app()
        state = _make_state(batcher or FakeBatcher(), model_version)
        app.dependency_overrides[get_state] = lambda: state
        return app

    return _factory


@pytest.fixture
def client(app_factory: Callable[..., FastAPI]) -> Iterator[TestClient]:
    with TestClient(app_factory()) as c:
        yield c


@pytest.fixture
def auth_headers() -> dict[str, str]:
    return {"x-api-key": API_KEY}
