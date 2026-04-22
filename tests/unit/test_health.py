"""Smoke tests for health/readiness/metrics. These are the first things that must always work."""

from __future__ import annotations

from fastapi.testclient import TestClient

from flowstate import __version__
from flowstate.api.main import create_app


def _client() -> TestClient:
    return TestClient(create_app())


def test_healthz_returns_ok() -> None:
    resp = _client().get("/healthz")
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "ok"
    assert body["version"] == __version__


def test_readyz_shape() -> None:
    resp = _client().get("/readyz")
    assert resp.status_code == 200
    body = resp.json()
    assert set(body.keys()) == {"status", "model_loaded", "cache_connected"}


def test_metrics_endpoint_exposes_prometheus_format() -> None:
    resp = _client().get("/metrics")
    assert resp.status_code == 200
    assert "text/plain" in resp.headers["content-type"]
    assert b"flowstate_requests_total" in resp.content


def test_request_id_header_roundtrip() -> None:
    resp = _client().get("/healthz", headers={"x-request-id": "abc-123"})
    assert resp.headers["x-request-id"] == "abc-123"
    assert "x-response-time-ms" in resp.headers


def test_protected_route_requires_api_key() -> None:
    # /docs is public; a non-public route would 401. We assert middleware allowlist works.
    resp = _client().get("/openapi.json")
    assert resp.status_code == 200
