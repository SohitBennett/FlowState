"""Load/reload of the inference subsystem.

Extracted from `api/main.py` so the `POST /admin/reload` route can reuse
the same code path the lifespan startup uses.
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

import structlog

from flowstate.api.deps import AppState
from flowstate.config import Settings
from flowstate.inference.batcher import DynamicBatcher
from flowstate.inference.runtime import ModelRuntime, RuntimeConfig
from flowstate.inference.tokenizer import CachedTokenizer
from flowstate.inference.warmup import warmup

log = structlog.get_logger(__name__)


def _resolve_model_version(onnx_path: Path) -> str:
    """Best-effort version string for the loaded model.

    Preference order:
      1. `<artifacts_root>/registry.json` written by `flowstate.training.register`
         (`<name>:<version>` from MLflow).
      2. The ONNX file's mtime epoch (`local:<epoch>`), so reloads are visible.
    """
    registry = onnx_path.parent.parent / "registry.json"
    if registry.exists():
        try:
            data = json.loads(registry.read_text())
            return f"{data['model_name']}:{data['version']}"
        except (KeyError, json.JSONDecodeError):
            pass
    return f"local:{int(onnx_path.stat().st_mtime)}"


async def load_inference(state: AppState, settings: Settings) -> None:
    """Load model + tokenizer + batcher into `state` and warm up.

    If the configured artifacts are missing, logs a warning and leaves the
    state empty so the API can still boot for development.
    """
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
    state.model_version = _resolve_model_version(onnx_path)
    state.model_loaded = True
    state.ready = True
    log.info(
        "inference_ready",
        labels=runtime.labels,
        model_version=state.model_version,
        onnx_path=str(onnx_path),
    )


async def reload_inference(state: AppState, settings: Settings) -> None:
    """Tear down the current inference subsystem and re-load from disk.

    In-flight requests on the old batcher are drained via `batcher.stop()`
    before the new runtime is loaded.
    """
    if state.reload_lock is None:
        state.reload_lock = asyncio.Lock()
    async with state.reload_lock:
        state.ready = False
        if state.batcher is not None:
            await state.batcher.stop()
        state.runtime = None
        state.tokenizer = None
        state.batcher = None
        state.model_loaded = False
        state.model_version = None
        await load_inference(state, settings)
