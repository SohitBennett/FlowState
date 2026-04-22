"""Prometheus metrics. Defined once, imported everywhere."""

from __future__ import annotations

from prometheus_client import CONTENT_TYPE_LATEST, Counter, Gauge, Histogram, generate_latest

REQUEST_COUNT = Counter(
    "flowstate_requests_total",
    "Total HTTP requests",
    ["method", "route", "status"],
)

REQUEST_LATENCY = Histogram(
    "flowstate_request_latency_seconds",
    "End-to-end request latency",
    ["route"],
    buckets=(0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5),
)

INFERENCE_LATENCY = Histogram(
    "flowstate_inference_latency_seconds",
    "Model inference latency (excluding queue + cache)",
    buckets=(0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25),
)

BATCH_SIZE = Histogram(
    "flowstate_batch_size",
    "Dynamic batch sizes flushed to the model",
    buckets=(1, 2, 4, 8, 16, 32, 64),
)

CACHE_HITS = Counter("flowstate_cache_hits_total", "Cache hits")
CACHE_MISSES = Counter("flowstate_cache_misses_total", "Cache misses")
QUEUE_DEPTH = Gauge("flowstate_batch_queue_depth", "Pending requests in batcher queue")


def render_metrics() -> tuple[bytes, str]:
    return generate_latest(), CONTENT_TYPE_LATEST
