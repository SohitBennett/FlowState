"""Integration tests for /v1/predict and /v1/predict/batch."""

from __future__ import annotations

from collections.abc import Callable

from fastapi import FastAPI
from fastapi.testclient import TestClient

from tests.integration.conftest import ExplodingBatcher, FakeBatcher


def test_predict_happy_path(client: TestClient, auth_headers: dict[str, str]) -> None:
    resp = client.post(
        "/v1/predict",
        json={"text": "Apple announces new products."},
        headers=auth_headers,
    )
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["model_version"] == "test:v1"
    assert body["cached"] is False
    assert body["prediction"]["label"] == "World"
    assert 0.0 <= body["prediction"]["score"] <= 1.0
    assert body["latency_ms"] >= 0
    assert "x-request-id" in resp.headers
    assert "x-response-time-ms" in resp.headers


def test_predict_batch_returns_one_prediction_per_input(
    client: TestClient, auth_headers: dict[str, str]
) -> None:
    resp = client.post(
        "/v1/predict/batch",
        json={"texts": ["a", "b", "c", "d"]},
        headers=auth_headers,
    )
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert len(body["predictions"]) == 4
    for p in body["predictions"]:
        assert "label" in p and "score" in p


def test_predict_extra_fields_rejected(
    client: TestClient, auth_headers: dict[str, str]
) -> None:
    resp = client.post(
        "/v1/predict",
        json={"text": "ok", "extra": "nope"},
        headers=auth_headers,
    )
    assert resp.status_code == 422


def test_predict_rejects_empty_text(
    client: TestClient, auth_headers: dict[str, str]
) -> None:
    resp = client.post("/v1/predict", json={"text": ""}, headers=auth_headers)
    assert resp.status_code == 422


def test_predict_rejects_text_over_max_length(
    client: TestClient, auth_headers: dict[str, str]
) -> None:
    resp = client.post(
        "/v1/predict",
        json={"text": "x" * 5000},
        headers=auth_headers,
    )
    assert resp.status_code == 422


def test_predict_batch_rejects_empty_list(
    client: TestClient, auth_headers: dict[str, str]
) -> None:
    resp = client.post("/v1/predict/batch", json={"texts": []}, headers=auth_headers)
    assert resp.status_code == 422


def test_predict_batch_rejects_oversized_list(
    client: TestClient, auth_headers: dict[str, str]
) -> None:
    resp = client.post(
        "/v1/predict/batch",
        json={"texts": ["x"] * 200},
        headers=auth_headers,
    )
    assert resp.status_code == 422


def test_predict_returns_503_when_model_not_loaded(
    app_factory: Callable[..., FastAPI], auth_headers: dict[str, str]
) -> None:
    from flowstate.api.deps import AppState, get_state

    app = app_factory()
    not_ready = AppState(model_loaded=False, ready=False)
    app.dependency_overrides[get_state] = lambda: not_ready
    with TestClient(app) as client:
        resp = client.post("/v1/predict", json={"text": "hi"}, headers=auth_headers)
    assert resp.status_code == 503
    assert resp.json()["error"]["code"] == "model_not_ready"


def test_predict_propagates_batcher_failure_as_500(
    app_factory: Callable[..., FastAPI], auth_headers: dict[str, str]
) -> None:
    app = app_factory(batcher=ExplodingBatcher())
    with TestClient(app) as client:
        resp = client.post("/v1/predict", json={"text": "hi"}, headers=auth_headers)
    assert resp.status_code == 500


def test_batcher_is_shared_across_routes(
    app_factory: Callable[..., FastAPI], auth_headers: dict[str, str]
) -> None:
    fake = FakeBatcher()
    app = app_factory(batcher=fake)
    with TestClient(app) as client:
        client.post("/v1/predict", json={"text": "a"}, headers=auth_headers)
        client.post(
            "/v1/predict/batch", json={"texts": ["b", "c", "d"]}, headers=auth_headers
        )
    assert fake.calls == 4
