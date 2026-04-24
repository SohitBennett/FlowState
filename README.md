# FlowState

> Enterprise MLOps & Real-Time Inference Engine — high-throughput text classification with full MLOps lifecycle.

**Targets:** p99 ≤ 50 ms · ≥ 500 RPS sustained · zero-downtime deploys · drift-aware · fully observable.

See [PLAN.md](PLAN.md) for the end-to-end delivery plan.

---

## Stack

FastAPI · ONNX Runtime (FP16) · Redis · MLflow · MinIO · Prometheus · Grafana · OpenTelemetry · GitHub Actions · Docker · Kubernetes/Helm

## Quickstart (local)

```bash
# 1. Install dev tooling
make dev

# 2. Bring up the full local stack (api, redis, mlflow, minio, postgres, prom, grafana)
cp .env.example .env
make up

# 3. Hit the API
curl -s http://localhost:8000/healthz
curl -s http://localhost:8000/metrics | head

# 4. Run tests
make test
```

| Service     | URL                     |
|-------------|-------------------------|
| API         | http://localhost:8000   |
| API docs    | http://localhost:8000/docs |
| MLflow      | http://localhost:5000   |
| MinIO       | http://localhost:9001   |
| Prometheus  | http://localhost:9090   |
| Grafana     | http://localhost:3000   |

## Training pipeline (Phase 1)

End-to-end: fine-tune → export to ONNX FP16 (with a cosine-similarity parity
gate) → register in MLflow under the `staging` alias.

```bash
# one-shot smoke run: train on 1k samples, 1 epoch, then export + register
make train    TRAIN_ARGS="--max-train-samples 1000 --epochs 1 --run-name smoke"
make export
make register

# full run (the one that matters)
make pipeline
```

Outputs:
- `artifacts/latest/model/` — fine-tuned HF model + tokenizer
- `artifacts/latest/eval_report.json` — accuracy, macro-F1, per-class F1, confusion matrix
- `artifacts/latest/onnx/model_fp16.onnx` — quantized, production-bound model
- `artifacts/latest/onnx/parity.json` — FP16 vs FP32 cosine similarity (gated ≥ 0.999)
- `artifacts/latest/registry.json` — MLflow model version + alias

MLflow UI at http://localhost:5000 logs all runs, params, metrics, and artifacts.

## Inference runtime (Phase 2)

The inference subsystem lives in [src/flowstate/inference/](src/flowstate/inference/):

- `runtime.py` — ONNX Runtime session with thread tuning, async offload via `asyncio.to_thread`.
- `tokenizer.py` — cached HF fast tokenizer with dynamic per-batch padding.
- `batcher.py` — asyncio dynamic micro-batcher (max_batch_size=32, max_wait_ms=5).
- `warmup.py` — synthetic inferences at varying sequence lengths, gates the readiness probe.
- `postprocess.py` — stable softmax + top-prediction.

The API lifespan loads the ONNX model (from `FLOWSTATE_MODEL_PATH`), starts the batcher,
warms up the session, and only then flips `/readyz` to `ok`. If artifacts are missing the
API still boots so development can proceed; `readyz` will report `model_loaded=false`.

### Phase 2 benchmark (exit gate)

```bash
make pipeline          # prereq: produces artifacts/latest/
make benchmark         # PyTorch FP32 vs ONNX FP32 vs ONNX FP16 + dynamic batcher
```

Writes `docs/perf/fp16-vs-fp32.md` + `.json`. Fails non-zero if the FP16 + batcher
throughput is less than 3x the naive PyTorch baseline.

## Project layout

See [PLAN.md §3](PLAN.md#3-repository-layout).

## Status

- [x] **Phase 0** — Foundations (tooling, compose stack, CI skeleton, FastAPI skeleton)
- [x] **Phase 1 (code)** — Training pipeline (AG News + DistilBERT + MLflow + ONNX FP16) — *awaiting first integration run*
- [x] **Phase 2 (code)** — Inference runtime (ONNX, dynamic batcher, warmup, benchmark harness) — *awaiting first benchmark run*
- [ ] Phase 3 — API layer (auth, rate limit, contract tests)
- [ ] Phase 4 — Redis caching (single-flight, jittered TTL)
- [ ] Phase 5 — Observability (metrics, logs, traces, dashboards, alerts)
- [ ] Phase 6 — CI/CD + zero-downtime deploys
- [ ] Phase 7 — Load testing & perf validation
- [ ] Phase 8 — Drift & continuous evaluation
- [ ] Phase 9 — Hardening + public VPS deploy
