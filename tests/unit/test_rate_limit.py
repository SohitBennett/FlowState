"""Token-bucket and RateLimiter tests with an injectable clock."""

from __future__ import annotations

import pytest

from flowstate.api.rate_limit import RateLimiter, TokenBucket


class _FakeClock:
    def __init__(self, start: float = 0.0) -> None:
        self._t = start

    def now(self) -> float:
        return self._t

    def advance(self, seconds: float) -> None:
        self._t += seconds


async def test_bucket_allows_burst_then_blocks() -> None:
    clock = _FakeClock()
    bucket = TokenBucket(rate=1.0, burst=5, time_fn=clock.now)
    for _ in range(5):
        assert await bucket.consume() is True
    assert await bucket.consume() is False


async def test_bucket_refills_at_configured_rate() -> None:
    clock = _FakeClock()
    bucket = TokenBucket(rate=10.0, burst=2, time_fn=clock.now)
    assert await bucket.consume() is True
    assert await bucket.consume() is True
    assert await bucket.consume() is False  # bucket empty
    clock.advance(0.05)  # 0.05s * 10 tokens/s = 0.5 tokens
    assert await bucket.consume() is False
    clock.advance(0.06)  # cumulative 0.11s -> 1.1 tokens (capped to burst=2)
    assert await bucket.consume() is True


async def test_bucket_caps_at_burst_size() -> None:
    clock = _FakeClock()
    bucket = TokenBucket(rate=100.0, burst=3, time_fn=clock.now)
    for _ in range(3):
        await bucket.consume()
    clock.advance(60.0)  # would refill to 6000 without cap
    successes = 0
    for _ in range(10):
        if await bucket.consume():
            successes += 1
    assert successes == 3  # bucket only refilled to `burst`


async def test_invalid_config_rejected() -> None:
    with pytest.raises(ValueError, match="rate"):
        TokenBucket(rate=0, burst=1)
    with pytest.raises(ValueError, match="burst"):
        TokenBucket(rate=1.0, burst=0)


async def test_rate_limiter_isolates_keys() -> None:
    clock = _FakeClock()
    limiter = RateLimiter(rate=1.0, burst=2, time_fn=clock.now)
    assert await limiter.check("alice") is True
    assert await limiter.check("alice") is True
    assert await limiter.check("alice") is False  # alice exhausted
    assert await limiter.check("bob") is True  # bob untouched
    assert limiter.tracked_keys == 2


async def test_rate_limiter_lazily_creates_buckets() -> None:
    clock = _FakeClock()
    limiter = RateLimiter(rate=1.0, burst=1, time_fn=clock.now)
    assert limiter.tracked_keys == 0
    await limiter.check("k1")
    await limiter.check("k2")
    await limiter.check("k1")
    assert limiter.tracked_keys == 2
