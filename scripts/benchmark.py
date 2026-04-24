"""Benchmark PyTorch FP32 vs ONNX FP32 vs ONNX FP16 + dynamic batching.

Exit gate for Phase 2: the `ONNX FP16 + batcher` throughput must be at
least 3x the `PyTorch naive` throughput. Writes a machine-specific
report to docs/perf/fp16-vs-fp32.md.

Usage:
    python scripts/benchmark.py \
        --artifacts artifacts/latest \
        --samples 500 --duration 15 --concurrency 32
"""

from __future__ import annotations

import argparse
import asyncio
import datetime as dt
import json
import platform
import statistics
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from flowstate.inference.batcher import DynamicBatcher
from flowstate.inference.runtime import ModelRuntime, RuntimeConfig
from flowstate.inference.tokenizer import CachedTokenizer

try:  # torch is an optional dep (train extras); benchmark degrades gracefully.
    import torch
    from transformers import AutoModelForSequenceClassification

    _HAS_TORCH = True
except ImportError:  # pragma: no cover
    _HAS_TORCH = False


@dataclass
class LatencyStats:
    mode: str
    p50_ms: float
    p95_ms: float
    p99_ms: float
    mean_ms: float
    n: int


@dataclass
class ThroughputStats:
    mode: str
    concurrency: int
    duration_s: float
    completed: int
    rps: float


def _latency_percentiles(mode: str, samples_ms: list[float]) -> LatencyStats:
    s = sorted(samples_ms)
    n = len(s)
    return LatencyStats(
        mode=mode,
        p50_ms=s[int(0.50 * (n - 1))],
        p95_ms=s[int(0.95 * (n - 1))],
        p99_ms=s[int(0.99 * (n - 1))],
        mean_ms=statistics.mean(s),
        n=n,
    )


def _sample_texts(n: int) -> list[str]:
    base = [
        "Markets rallied on strong earnings reports from major tech companies.",
        "The home team clinched the championship in overtime after a dramatic rally.",
        "Scientists announced a breakthrough in quantum error correction this week.",
        "Diplomatic tensions eased following a surprise summit between the two nations.",
    ]
    return [base[i % len(base)] for i in range(n)]


def _torch_baseline_latency(
    tokenizer: CachedTokenizer, model_dir: Path, texts: list[str]
) -> LatencyStats:
    model = AutoModelForSequenceClassification.from_pretrained(str(model_dir))
    model.eval()
    with torch.inference_mode():
        # burn-in
        for t in texts[:5]:
            enc = tokenizer.encode([t])
            model(
                input_ids=torch.from_numpy(enc["input_ids"]),
                attention_mask=torch.from_numpy(enc["attention_mask"]),
            )
        times = []
        for t in texts:
            enc = tokenizer.encode([t])
            start = time.perf_counter()
            model(
                input_ids=torch.from_numpy(enc["input_ids"]),
                attention_mask=torch.from_numpy(enc["attention_mask"]),
            )
            times.append((time.perf_counter() - start) * 1000)
    return _latency_percentiles("PyTorch FP32 (single)", times)


def _onnx_single_sample_latency(
    runtime: ModelRuntime, tokenizer: CachedTokenizer, label: str, texts: list[str]
) -> LatencyStats:
    # burn-in
    for t in texts[:5]:
        enc = tokenizer.encode([t])
        runtime.run(enc["input_ids"], enc["attention_mask"])
    times = []
    for t in texts:
        enc = tokenizer.encode([t])
        start = time.perf_counter()
        runtime.run(enc["input_ids"], enc["attention_mask"])
        times.append((time.perf_counter() - start) * 1000)
    return _latency_percentiles(label, times)


def _torch_baseline_throughput(
    tokenizer: CachedTokenizer,
    model_dir: Path,
    duration_s: float,
    corpus: list[str],
) -> ThroughputStats:
    model = AutoModelForSequenceClassification.from_pretrained(str(model_dir))
    model.eval()
    completed = 0
    deadline = time.perf_counter() + duration_s
    idx = 0
    with torch.inference_mode():
        while time.perf_counter() < deadline:
            enc = tokenizer.encode([corpus[idx % len(corpus)]])
            model(
                input_ids=torch.from_numpy(enc["input_ids"]),
                attention_mask=torch.from_numpy(enc["attention_mask"]),
            )
            completed += 1
            idx += 1
    elapsed = duration_s
    return ThroughputStats(
        mode="PyTorch FP32 naive loop",
        concurrency=1,
        duration_s=elapsed,
        completed=completed,
        rps=completed / elapsed,
    )


async def _batched_throughput(
    batcher: DynamicBatcher,
    concurrency: int,
    duration_s: float,
    corpus: list[str],
    label: str,
) -> ThroughputStats:
    stop_at = time.perf_counter() + duration_s
    completed = 0
    counter_lock = asyncio.Lock()

    async def worker(worker_id: int) -> None:
        nonlocal completed
        idx = worker_id
        while time.perf_counter() < stop_at:
            await batcher.submit(corpus[idx % len(corpus)])
            async with counter_lock:
                completed += 1
            idx += concurrency

    await asyncio.gather(*(worker(i) for i in range(concurrency)))
    return ThroughputStats(
        mode=label,
        concurrency=concurrency,
        duration_s=duration_s,
        completed=completed,
        rps=completed / duration_s,
    )


def _render_markdown(
    latency: list[LatencyStats],
    throughput: list[ThroughputStats],
    meta: dict[str, str],
    speedup: float | None,
    target: float,
    passed: bool | None,
) -> str:
    lines = [
        "# FlowState — Inference Runtime Benchmarks",
        "",
        f"Generated: {meta['generated_at']}  ",
        f"Host: {meta['host']}  ",
        f"Python: {meta['python']}  ",
        f"ONNX Runtime: {meta['onnxruntime']}  ",
        f"Torch: {meta['torch']}  ",
        "",
        "## Single-sample latency (ms)",
        "",
        "| Mode | n | p50 | p95 | p99 | mean |",
        "|------|---|-----|-----|-----|------|",
    ]
    for s in latency:
        lines.append(
            f"| {s.mode} | {s.n} | {s.p50_ms:.2f} | {s.p95_ms:.2f} |"
            f" {s.p99_ms:.2f} | {s.mean_ms:.2f} |"
        )
    lines += [
        "",
        "## Throughput (requests/second)",
        "",
        "| Mode | concurrency | duration (s) | completed | RPS |",
        "|------|-------------|--------------|-----------|-----|",
    ]
    for t in throughput:
        lines.append(
            f"| {t.mode} | {t.concurrency} | {t.duration_s:.1f} |"
            f" {t.completed} | {t.rps:.1f} |"
        )
    lines += [
        "",
        "## Phase 2 exit gate",
        "",
        f"Target: ONNX FP16 + batcher throughput >= {target:g}x PyTorch naive baseline.",
    ]
    if speedup is not None:
        verdict = "PASS" if passed else "FAIL"
        lines.append(f"Observed speedup: **{speedup:.2f}x** — **{verdict}**")
    else:
        lines.append("Speedup: not measured (PyTorch unavailable).")
    lines += ["", "Regenerate with: `make benchmark`.", ""]
    return "\n".join(lines)


async def _run(args: argparse.Namespace) -> int:
    artifacts = Path(args.artifacts)
    onnx_fp16 = artifacts / "onnx" / "model_fp16.onnx"
    onnx_fp32 = artifacts / "onnx" / "model_fp32.onnx"
    tokenizer_dir = artifacts / "model"
    if not onnx_fp16.exists() or not tokenizer_dir.exists():
        raise SystemExit(
            f"missing artifacts; expected {onnx_fp16} and {tokenizer_dir}. "
            "Run `make pipeline` first."
        )

    tokenizer = CachedTokenizer(tokenizer_dir, max_seq_len=128)
    runtime_cfg = RuntimeConfig()

    fp16_runtime = ModelRuntime.from_artifacts(onnx_fp16, tokenizer_dir, runtime_cfg)

    corpus = _sample_texts(args.samples)

    latency: list[LatencyStats] = []
    if _HAS_TORCH:
        latency.append(_torch_baseline_latency(tokenizer, tokenizer_dir, corpus))
    if onnx_fp32.exists():
        fp32_runtime = ModelRuntime.from_artifacts(
            onnx_fp32, tokenizer_dir, runtime_cfg
        )
        latency.append(
            _onnx_single_sample_latency(
                fp32_runtime, tokenizer, "ONNX FP32 (single)", corpus
            )
        )
    latency.append(
        _onnx_single_sample_latency(
            fp16_runtime, tokenizer, "ONNX FP16 (single)", corpus
        )
    )

    batcher = DynamicBatcher(tokenizer, fp16_runtime, max_batch_size=32, max_wait_ms=5)
    await batcher.start()
    try:
        fp16_tput = await _batched_throughput(
            batcher,
            concurrency=args.concurrency,
            duration_s=args.duration,
            corpus=corpus,
            label="ONNX FP16 + dynamic batcher",
        )
    finally:
        await batcher.stop()

    throughput: list[ThroughputStats] = []
    torch_tput: ThroughputStats | None = None
    if _HAS_TORCH:
        torch_tput = _torch_baseline_throughput(
            tokenizer, tokenizer_dir, duration_s=args.duration, corpus=corpus
        )
        throughput.append(torch_tput)
    throughput.append(fp16_tput)

    speedup = (fp16_tput.rps / torch_tput.rps) if torch_tput else None
    target = args.target
    passed = (speedup >= target) if speedup is not None else None

    import onnxruntime as ort

    meta = {
        "generated_at": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
        "host": f"{platform.system()} {platform.machine()} ({platform.processor() or 'cpu'})",
        "python": platform.python_version(),
        "onnxruntime": ort.__version__,
        "torch": torch.__version__ if _HAS_TORCH else "not installed",
    }

    report = _render_markdown(latency, throughput, meta, speedup, target, passed)
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(report)

    summary = {
        "latency": [asdict(s) for s in latency],
        "throughput": [asdict(t) for t in throughput],
        "speedup_vs_torch_naive": speedup,
        "target_speedup": target,
        "passed": passed,
        "meta": meta,
    }
    (out_path.parent / "fp16-vs-fp32.json").write_text(json.dumps(summary, indent=2))

    print(report)
    if passed is False:
        print(f"[benchmark] FAIL: speedup {speedup:.2f}x < target {target:g}x")
        return 1
    return 0


def main() -> None:
    p = argparse.ArgumentParser(description="FlowState inference benchmark.")
    p.add_argument("--artifacts", default="artifacts/latest")
    p.add_argument("--samples", type=int, default=500, help="single-sample latency N")
    p.add_argument("--duration", type=float, default=15.0, help="throughput window (s)")
    p.add_argument("--concurrency", type=int, default=32)
    p.add_argument("--target", type=float, default=3.0, help="required speedup vs torch")
    p.add_argument("--output", default="docs/perf/fp16-vs-fp32.md")
    ns = p.parse_args()
    np.random.seed(0)
    exit_code = asyncio.run(_run(ns))
    raise SystemExit(exit_code)


if __name__ == "__main__":
    main()
