"""Asyncio dynamic micro-batching.

A single background worker pulls pending requests off an asyncio.Queue and
flushes a batch when EITHER of these thresholds is reached:
    * `max_batch_size` requests are queued, OR
    * `max_wait_ms` has elapsed since the first request in the batch.

This is the single biggest throughput lever for transformer inference on
CPU: tokenization and model execution amortize across requests, so a batch
of 32 is typically 5–20x cheaper per-sample than 32 sequential calls.

The worker is resilient to per-batch failures: a model or tokenizer
exception is propagated to the callers of that batch via their futures,
and the worker continues serving subsequent batches.
"""

from __future__ import annotations

import asyncio
import contextlib
import time
from dataclasses import dataclass

import structlog

from flowstate.inference.postprocess import ClassificationResult, top_prediction
from flowstate.inference.runtime import ModelRuntime
from flowstate.inference.tokenizer import CachedTokenizer
from flowstate.monitoring.metrics import (
    BATCH_SIZE,
    INFERENCE_LATENCY,
    QUEUE_DEPTH,
)

log = structlog.get_logger(__name__)


@dataclass
class _PendingRequest:
    text: str
    future: asyncio.Future[ClassificationResult]


class DynamicBatcher:
    def __init__(
        self,
        tokenizer: CachedTokenizer,
        runtime: ModelRuntime,
        max_batch_size: int = 32,
        max_wait_ms: int = 5,
    ) -> None:
        if max_batch_size < 1:
            raise ValueError("max_batch_size must be >= 1")
        if max_wait_ms < 0:
            raise ValueError("max_wait_ms must be >= 0")
        self._tokenizer = tokenizer
        self._runtime = runtime
        self._max_batch_size = max_batch_size
        self._max_wait_s = max_wait_ms / 1000.0
        self._queue: asyncio.Queue[_PendingRequest] = asyncio.Queue()
        self._worker_task: asyncio.Task[None] | None = None
        self._stopping = False

    async def start(self) -> None:
        if self._worker_task is not None:
            return
        self._worker_task = asyncio.create_task(self._run(), name="flowstate-batcher")

    async def stop(self, drain_timeout_s: float = 10.0) -> None:
        """Stop accepting new work, drain the queue, then tear down the worker."""
        self._stopping = True
        deadline = time.monotonic() + drain_timeout_s
        while not self._queue.empty() and time.monotonic() < deadline:
            await asyncio.sleep(0.01)
        if self._worker_task is not None and not self._worker_task.done():
            self._worker_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._worker_task
        # Fail any stragglers that slipped in during shutdown.
        while not self._queue.empty():
            req = self._queue.get_nowait()
            if not req.future.done():
                req.future.set_exception(RuntimeError("batcher stopped"))

    async def submit(self, text: str) -> ClassificationResult:
        if self._stopping:
            raise RuntimeError("batcher is stopping")
        if self._worker_task is None:
            raise RuntimeError("batcher not started")
        fut: asyncio.Future[ClassificationResult] = (
            asyncio.get_running_loop().create_future()
        )
        await self._queue.put(_PendingRequest(text=text, future=fut))
        QUEUE_DEPTH.set(self._queue.qsize())
        return await fut

    async def _run(self) -> None:
        while not self._stopping:
            # Short timeout on the initial wait lets us notice shutdown promptly
            # without burning CPU; once we have a first item, we complete the
            # batch uninterrupted.
            try:
                first = await asyncio.wait_for(self._queue.get(), timeout=0.1)
            except asyncio.TimeoutError:
                continue

            batch: list[_PendingRequest] = [first]
            deadline = time.monotonic() + self._max_wait_s
            while len(batch) < self._max_batch_size:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    break
                try:
                    item = await asyncio.wait_for(self._queue.get(), timeout=remaining)
                    batch.append(item)
                except asyncio.TimeoutError:
                    break

            await self._process_batch(batch)

    async def _process_batch(self, batch: list[_PendingRequest]) -> None:
        BATCH_SIZE.observe(len(batch))
        QUEUE_DEPTH.set(self._queue.qsize())
        texts = [r.text for r in batch]
        try:
            start = time.perf_counter()
            enc = self._tokenizer.encode(texts)
            logits = await self._runtime.run_async(
                enc["input_ids"], enc["attention_mask"]
            )
            INFERENCE_LATENCY.observe(time.perf_counter() - start)
            results = top_prediction(logits, self._runtime.labels)
            for req, result in zip(batch, results, strict=True):
                if not req.future.done():
                    req.future.set_result(result)
        except Exception as exc:  # noqa: BLE001 — propagate to all callers
            log.exception("batcher_inference_failed", batch_size=len(batch))
            for req in batch:
                if not req.future.done():
                    req.future.set_exception(exc)
