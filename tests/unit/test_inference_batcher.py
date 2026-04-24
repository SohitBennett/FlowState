"""Tests for the dynamic async batcher using a fake runtime + tokenizer.

We avoid loading a real ONNX model so these remain fast pure-unit tests.
"""

from __future__ import annotations

import asyncio

import numpy as np
import pytest

from flowstate.inference.batcher import DynamicBatcher


class _FakeTokenizer:
    def encode(self, texts: list[str]) -> dict[str, np.ndarray]:
        n = len(texts)
        return {
            "input_ids": np.zeros((n, 8), dtype=np.int64),
            "attention_mask": np.ones((n, 8), dtype=np.int64),
        }


class _FakeRuntime:
    def __init__(self, labels: list[str], delay_s: float = 0.0) -> None:
        self._labels = labels
        self._delay_s = delay_s
        self.batch_sizes: list[int] = []
        self.calls = 0

    @property
    def labels(self) -> list[str]:
        return self._labels

    def _logits(self, batch_size: int) -> np.ndarray:
        logits = np.full((batch_size, len(self._labels)), -1.0, dtype=np.float32)
        logits[:, 0] = 5.0  # pin argmax to label[0]
        return logits

    async def run_async(
        self, input_ids: np.ndarray, attention_mask: np.ndarray
    ) -> np.ndarray:
        self.calls += 1
        self.batch_sizes.append(int(input_ids.shape[0]))
        if self._delay_s:
            await asyncio.sleep(self._delay_s)
        return self._logits(input_ids.shape[0])


async def _new_batcher(
    max_batch_size: int = 4, max_wait_ms: int = 20, delay_s: float = 0.0
) -> tuple[DynamicBatcher, _FakeRuntime]:
    runtime = _FakeRuntime(labels=["world", "sports"], delay_s=delay_s)
    batcher = DynamicBatcher(
        tokenizer=_FakeTokenizer(),
        runtime=runtime,  # type: ignore[arg-type]
        max_batch_size=max_batch_size,
        max_wait_ms=max_wait_ms,
    )
    await batcher.start()
    return batcher, runtime


async def test_single_request_resolves_with_label_and_score() -> None:
    batcher, _ = await _new_batcher()
    try:
        result = await batcher.submit("hello")
        assert result.label == "world"
        assert 0.0 < result.score <= 1.0
    finally:
        await batcher.stop()


async def test_concurrent_requests_are_batched() -> None:
    batcher, runtime = await _new_batcher(max_batch_size=8, max_wait_ms=30)
    try:
        results = await asyncio.gather(*(batcher.submit(f"req-{i}") for i in range(8)))
        assert len(results) == 8
        # With 8 concurrent submits and max_batch_size=8, one batch should absorb them all.
        assert max(runtime.batch_sizes) >= 2
        assert sum(runtime.batch_sizes) == 8
    finally:
        await batcher.stop()


async def test_max_batch_size_is_respected() -> None:
    batcher, runtime = await _new_batcher(max_batch_size=3, max_wait_ms=50)
    try:
        results = await asyncio.gather(*(batcher.submit(f"req-{i}") for i in range(10)))
        assert len(results) == 10
        assert all(bs <= 3 for bs in runtime.batch_sizes)
        assert sum(runtime.batch_sizes) == 10
    finally:
        await batcher.stop()


async def test_max_wait_flushes_single_request_quickly() -> None:
    batcher, _ = await _new_batcher(max_batch_size=32, max_wait_ms=10)
    try:
        start = asyncio.get_running_loop().time()
        await batcher.submit("solo")
        elapsed_ms = (asyncio.get_running_loop().time() - start) * 1000
        # Poll-tick is 100ms in _run(); add generous slack for CI jitter.
        assert elapsed_ms < 500
    finally:
        await batcher.stop()


async def test_submit_after_stop_raises() -> None:
    batcher, _ = await _new_batcher()
    await batcher.stop()
    with pytest.raises(RuntimeError, match="stopping"):
        await batcher.submit("too late")


async def test_submit_without_start_raises() -> None:
    batcher = DynamicBatcher(
        tokenizer=_FakeTokenizer(),
        runtime=_FakeRuntime(labels=["a"]),  # type: ignore[arg-type]
    )
    with pytest.raises(RuntimeError, match="not started"):
        await batcher.submit("nope")


async def test_inference_exception_is_propagated_to_callers() -> None:
    class _BrokenRuntime(_FakeRuntime):
        async def run_async(self, *_: object, **__: object) -> np.ndarray:
            raise RuntimeError("model broken")

    batcher = DynamicBatcher(
        tokenizer=_FakeTokenizer(),
        runtime=_BrokenRuntime(labels=["a"]),  # type: ignore[arg-type]
    )
    await batcher.start()
    try:
        with pytest.raises(RuntimeError, match="model broken"):
            await batcher.submit("x")
    finally:
        await batcher.stop()


async def test_invalid_config_rejected() -> None:
    with pytest.raises(ValueError, match="max_batch_size"):
        DynamicBatcher(
            tokenizer=_FakeTokenizer(),
            runtime=_FakeRuntime(labels=["a"]),  # type: ignore[arg-type]
            max_batch_size=0,
        )
    with pytest.raises(ValueError, match="max_wait_ms"):
        DynamicBatcher(
            tokenizer=_FakeTokenizer(),
            runtime=_FakeRuntime(labels=["a"]),  # type: ignore[arg-type]
            max_wait_ms=-1,
        )
