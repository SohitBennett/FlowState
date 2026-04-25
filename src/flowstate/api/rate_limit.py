"""In-memory token-bucket rate limiter, keyed by API key.

Each key gets a bucket that refills at `rate` tokens/second up to a maximum
of `burst` tokens. Each request consumes one token; if none are available
the request is denied with 429.

This is process-local — under multiple gunicorn workers the effective rate
is `workers * rate`. Phase 4 may move the bucket state into Redis to share
limits across workers; for Phase 3 this is sufficient.
"""

from __future__ import annotations

import asyncio
import time
from collections.abc import Callable


class TokenBucket:
    """A single token bucket. Time source is injectable for deterministic tests."""

    def __init__(
        self,
        rate: float,
        burst: int,
        time_fn: Callable[[], float] = time.monotonic,
    ) -> None:
        if rate <= 0:
            raise ValueError("rate must be > 0")
        if burst < 1:
            raise ValueError("burst must be >= 1")
        self._rate = rate
        self._burst = float(burst)
        self._time_fn = time_fn
        self._tokens = float(burst)
        self._last = time_fn()
        self._lock = asyncio.Lock()

    async def consume(self, n: int = 1) -> bool:
        async with self._lock:
            now = self._time_fn()
            elapsed = max(0.0, now - self._last)
            self._tokens = min(self._burst, self._tokens + elapsed * self._rate)
            self._last = now
            if self._tokens >= n:
                self._tokens -= n
                return True
            return False


class RateLimiter:
    """Manages one TokenBucket per key, lazily allocated on first use."""

    def __init__(
        self,
        rate: float,
        burst: int,
        time_fn: Callable[[], float] = time.monotonic,
    ) -> None:
        self._rate = rate
        self._burst = burst
        self._time_fn = time_fn
        self._buckets: dict[str, TokenBucket] = {}
        self._lock = asyncio.Lock()

    async def check(self, key: str) -> bool:
        async with self._lock:
            bucket = self._buckets.get(key)
            if bucket is None:
                bucket = TokenBucket(self._rate, self._burst, self._time_fn)
                self._buckets[key] = bucket
        return await bucket.consume()

    @property
    def tracked_keys(self) -> int:
        return len(self._buckets)
