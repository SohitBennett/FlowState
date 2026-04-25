"""Schemathesis fuzz: assert the API never crashes (no 5xx) on random inputs.

Schemathesis loads the OpenAPI spec from the FastAPI app via ASGI (no live
server) and generates valid + edge-case inputs for every documented
endpoint. We override `get_state` with a deterministic fake batcher so
predict routes can return real 200 responses rather than 503s.
"""

from __future__ import annotations

import schemathesis

from flowstate.api.deps import AppState, get_state
from flowstate.api.main import create_app
from flowstate.inference.postprocess import ClassificationResult


_API_KEY = "schemathesis-test-key"


class _FakeBatcher:
    async def submit(self, _: str) -> ClassificationResult:
        return ClassificationResult(label="World", score=0.9)

    async def stop(self, *_: object, **__: object) -> None:
        return None


def _build_app():
    import os

    os.environ["FLOWSTATE_API_KEY"] = _API_KEY
    from flowstate.config import get_settings

    get_settings.cache_clear()

    app = create_app()
    state = AppState(
        batcher=_FakeBatcher(),  # type: ignore[arg-type]
        model_loaded=True,
        ready=True,
        model_version="schemathesis:v1",
    )
    app.dependency_overrides[get_state] = lambda: state
    return app


schema = schemathesis.openapi.from_asgi("/openapi.json", _build_app())


@schema.parametrize()
def test_no_5xx_on_random_inputs(case: schemathesis.Case) -> None:
    response = case.call(headers={"x-api-key": _API_KEY})
    assert response.status_code < 500, (
        f"{case.method} {case.path} -> {response.status_code}: {response.text[:300]}"
    )
