"""FastAPI application factory.

Lifespan hook loads the ONNX runtime, starts the dynamic batcher, and
runs warmup before readiness flips green. If the model artifacts are
missing (fresh clone, pre-training), the app still boots but readyz
reports `model_loaded=False` so the API can be developed iteratively.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import ORJSONResponse

from flowstate import __version__
from flowstate.api.deps import AppState
from flowstate.api.errors import FlowStateError, flowstate_error_handler
from flowstate.api.middleware import AuthMiddleware, RequestContextMiddleware
from flowstate.api.routes import health
from flowstate.config import Settings, get_settings
from flowstate.inference.batcher import DynamicBatcher
from flowstate.inference.runtime import ModelRuntime, RuntimeConfig
from flowstate.inference.tokenizer import CachedTokenizer
from flowstate.inference.warmup import warmup
from flowstate.logging import configure_logging, get_logger


async def _try_load_inference(state: AppState, settings: Settings) -> None:
    log = get_logger(__name__)
    onnx_path = Path(settings.model_path)
    tokenizer_dir = Path(settings.tokenizer_path)

    if not onnx_path.exists() or not tokenizer_dir.exists():
        log.warning(
            "model_artifacts_missing",
            onnx_path=str(onnx_path),
            tokenizer_dir=str(tokenizer_dir),
            hint="run `make pipeline` to produce artifacts",
        )
        return

    runtime = ModelRuntime.from_artifacts(
        onnx_path=onnx_path,
        tokenizer_dir=tokenizer_dir,
        runtime_cfg=RuntimeConfig(
            intra_op_num_threads=settings.intra_op_num_threads,
            inter_op_num_threads=settings.inter_op_num_threads,
        ),
    )
    tokenizer = CachedTokenizer(tokenizer_dir, max_seq_len=settings.max_seq_len)
    batcher = DynamicBatcher(
        tokenizer=tokenizer,
        runtime=runtime,
        max_batch_size=settings.batch_max_size,
        max_wait_ms=settings.batch_max_wait_ms,
    )
    await batcher.start()
    await warmup(runtime, tokenizer, iterations=settings.warmup_iterations)

    state.runtime = runtime
    state.tokenizer = tokenizer
    state.batcher = batcher
    state.model_loaded = True
    state.ready = True
    log.info("inference_ready", labels=runtime.labels, onnx_path=str(onnx_path))


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    settings = get_settings()
    configure_logging(settings.log_level)
    log = get_logger(__name__)
    log.info("flowstate_starting", env=settings.env, version=__version__)

    state: AppState = app.state.flowstate

    try:
        await _try_load_inference(state, settings)
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
    app = FastAPI(
        title="FlowState Inference API",
        version=__version__,
        default_response_class=ORJSONResponse,
        lifespan=lifespan,
    )

    # Seed the per-app state eagerly so endpoints that depend on it work even
    # when lifespan hasn't run (e.g. in tests that construct TestClient without
    # using it as a context manager). Lifespan later mutates this instance.
    app.state.flowstate = AppState()

    app.add_middleware(AuthMiddleware)
    app.add_middleware(RequestContextMiddleware)

    app.add_exception_handler(FlowStateError, flowstate_error_handler)  # type: ignore[arg-type]

    app.include_router(health.router)
    return app


app = create_app()
