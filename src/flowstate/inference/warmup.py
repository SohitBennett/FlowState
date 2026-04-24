"""Warmup the inference runtime before the readiness probe flips green.

ONNX Runtime lazily allocates buffers and primes CPU caches on the first
few forward passes; serving real traffic before warmup completes yields
cold-start latency spikes on p99. We run `n` synthetic inferences at
varying sequence lengths so subsequent real requests hit a hot path.
"""

from __future__ import annotations

import time

import structlog

from flowstate.inference.runtime import ModelRuntime
from flowstate.inference.tokenizer import CachedTokenizer

log = structlog.get_logger(__name__)


def _warmup_texts(n: int) -> list[str]:
    short = "ok"
    medium = " ".join(["warmup"] * 16)
    long = " ".join(["warmup"] * 64)
    cycle = [short, medium, long]
    return [cycle[i % 3] for i in range(n)]


async def warmup(
    runtime: ModelRuntime, tokenizer: CachedTokenizer, iterations: int = 5
) -> float:
    """Run `iterations` synthetic single-sample inferences. Returns total seconds."""
    if iterations <= 0:
        return 0.0
    start = time.perf_counter()
    for text in _warmup_texts(iterations):
        enc = tokenizer.encode([text])
        await runtime.run_async(enc["input_ids"], enc["attention_mask"])
    elapsed = time.perf_counter() - start
    log.info("warmup_complete", iterations=iterations, seconds=elapsed)
    return elapsed
