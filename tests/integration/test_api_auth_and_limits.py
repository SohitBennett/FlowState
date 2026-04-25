"""Integration tests for auth, rate limiting, and request size limits."""

from __future__ import annotations

from collections.abc import Callable

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from flowstate.config import get_settings


def test_predict_requires_api_key(client: TestClient) -> None:
    resp = client.post("/v1/predict", json={"text": "hi"})
    assert resp.status_code == 401
    assert resp.json()["error"]["code"] == "unauthorized"


def test_predict_rejects_wrong_api_key(client: TestClient) -> None:
    resp = client.post(
        "/v1/predict",
        json={"text": "hi"},
        headers={"x-api-key": "definitely-wrong"},
    )
    assert resp.status_code == 401


def test_public_endpoints_do_not_require_api_key(client: TestClient) -> None:
    for path in ("/healthz", "/readyz", "/metrics", "/openapi.json"):
        resp = client.get(path)
        assert resp.status_code == 200, f"{path} returned {resp.status_code}"


def test_admin_reload_requires_api_key(client: TestClient) -> None:
    resp = client.post("/admin/reload")
    assert resp.status_code == 401


def test_rate_limit_returns_429_after_burst(
    monkeypatch: pytest.MonkeyPatch,
    app_factory: Callable[..., FastAPI],
    auth_headers: dict[str, str],
) -> None:
    monkeypatch.setenv("FLOWSTATE_RATE_LIMIT_RATE", "1")
    monkeypatch.setenv("FLOWSTATE_RATE_LIMIT_BURST", "3")
    get_settings.cache_clear()

    with TestClient(app_factory()) as client:
        statuses = [
            client.post("/v1/predict", json={"text": "hi"}, headers=auth_headers).status_code
            for _ in range(6)
        ]
    # First 3 burst tokens succeed, the next requests are rate-limited.
    assert statuses[:3] == [200, 200, 200]
    assert 429 in statuses[3:]


def test_rate_limit_429_uses_standard_error_envelope(
    monkeypatch: pytest.MonkeyPatch, app_factory: Callable[..., FastAPI]
) -> None:
    # Per-key isolation is covered by the unit tests in test_rate_limit.py
    # (auth admits exactly one configured key, so we can't exercise multiple
    # keys end-to-end). Here we only verify the 429 path and error envelope.
    monkeypatch.setenv("FLOWSTATE_RATE_LIMIT_RATE", "1")
    monkeypatch.setenv("FLOWSTATE_RATE_LIMIT_BURST", "1")
    get_settings.cache_clear()

    headers = {"x-api-key": "test-key-for-integration-tests"}
    with TestClient(app_factory()) as client:
        first = client.post("/v1/predict", json={"text": "x"}, headers=headers)
        second = client.post("/v1/predict", json={"text": "x"}, headers=headers)

    assert first.status_code == 200
    assert second.status_code == 429
    assert second.json()["error"]["code"] == "rate_limited"


def test_admin_reload_succeeds_without_artifacts(
    client: TestClient, auth_headers: dict[str, str]
) -> None:
    # No artifacts on disk in tests, so reload should resolve to a not-ready
    # state but still return 200 with the standard envelope (ready=False).
    resp = client.post("/admin/reload", headers=auth_headers)
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["status"] == "reloaded"
    assert body["ready"] is False


def test_request_too_large_returns_413(
    monkeypatch: pytest.MonkeyPatch,
    app_factory: Callable[..., FastAPI],
    auth_headers: dict[str, str],
) -> None:
    monkeypatch.setenv("FLOWSTATE_MAX_REQUEST_BYTES", "200")
    get_settings.cache_clear()

    big_payload = {"texts": ["x" * 50] * 5}  # well over 200 bytes once serialized
    with TestClient(app_factory()) as client:
        resp = client.post("/v1/predict/batch", json=big_payload, headers=auth_headers)
    assert resp.status_code == 413
    assert resp.json()["error"]["code"] == "request_too_large"


def test_request_id_is_echoed_when_supplied(
    client: TestClient, auth_headers: dict[str, str]
) -> None:
    resp = client.post(
        "/v1/predict",
        json={"text": "hi"},
        headers={**auth_headers, "x-request-id": "trace-abc-123"},
    )
    assert resp.headers["x-request-id"] == "trace-abc-123"
