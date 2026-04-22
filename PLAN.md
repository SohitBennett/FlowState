# FlowState — Enterprise MLOps & Real-Time Inference Engine

> Production-grade, high-throughput ML inference platform for real-time text classification.
> Not a learning toy: every component is built to the standard of what would actually ship at a mid-to-large tech company.

---

## 1. Vision & Success Criteria

**Mission:** Serve a fine-tuned transformer (text classification) over HTTP/gRPC at **p99 < 50 ms** and **≥ 500 RPS sustained** on a single mid-tier node, with full MLOps lifecycle (train → register → deploy → monitor → rollback) and **zero-downtime updates**.

**Definition of Done (production bar):**
- [ ] Reproducible training pipeline (seeded, versioned data + code + model).
- [ ] Model registry with stage transitions (Staging → Production → Archived).
- [ ] Inference API with auth, rate limiting, observability, graceful shutdown, health/readiness probes.
- [ ] Redis caching layer with TTL + cache stampede protection.
- [ ] FP16 quantized model + dynamic batching.
- [ ] CI/CD: lint → unit → integration → load test → container build → security scan → deploy.
- [ ] Blue/green or canary deploys behind a load balancer (zero-downtime).
- [ ] Metrics (Prometheus), logs (structured JSON), traces (OpenTelemetry), dashboards (Grafana).
- [ ] Drift + latency + error-rate alerts.
- [ ] Documented SLOs, runbooks, and rollback procedures.
- [ ] Load test report proving the 500 RPS / 65% latency reduction claims.

---

## 2. High-Level Architecture

```
                    ┌─────────────┐
   Client ──TLS──▶  │   NGINX /   │  ──▶  ┌──────────────────┐
                    │   Traefik   │       │  FastAPI Inference│ ──▶ Redis (cache)
                    │  (LB + TLS) │       │   (uvicorn+gunicorn)│
                    └─────────────┘       │   - Auth (JWT/API)│ ──▶ Model Runtime
                          │               │   - Rate limit    │     (ONNX Runtime
                          │               │   - Batching      │      / Triton, FP16)
                          ▼               │   - OTel tracing  │
                    Prometheus  ◀─────────┤   - /metrics      │
                          │               └──────────────────┘
                          ▼                        │
                       Grafana                     ▼
                                              MLflow Registry
                                                   │
                                                   ▼
                                        S3 / MinIO (artifacts)
```

**Core stack**
- **Serving:** FastAPI + Uvicorn workers behind Gunicorn; ONNX Runtime (FP16) for inference; optional NVIDIA Triton for GPU path.
- **Model:** Fine-tuned `distilbert-base-uncased` (or `roberta-base`) on a public classification dataset (e.g., AG News, SST-2, or Jigsaw toxicity).
- **Cache:** Redis 7 (LRU, TTL, single-flight via `SETNX` lock).
- **Registry & tracking:** MLflow + MinIO (S3-compatible) backing store, Postgres metadata.
- **Orchestration:** Docker + docker-compose (local/dev) → Kubernetes manifests + Helm chart (prod).
- **CI/CD:** GitHub Actions (test, build, scan, push, deploy), with `workflow_dispatch` for promotion.
- **Observability:** Prometheus, Grafana, Loki (logs), Tempo/Jaeger (traces), OpenTelemetry SDK.
- **Load testing:** Locust + k6 (two tools, cross-validated).
- **Security:** Trivy (image scan), Bandit (Python SAST), pip-audit, SBOM via Syft.

---

## 3. Repository Layout

```
FlowState/
├── README.md
├── PLAN.md
├── LICENSE
├── Makefile
├── pyproject.toml              # ruff, black, mypy, pytest config
├── .pre-commit-config.yaml
├── .github/
│   └── workflows/
│       ├── ci.yml              # lint + test + scan on PR
│       ├── cd.yml              # build + push + deploy on main / tag
│       └── load-test.yml       # nightly perf regression
├── docker/
│   ├── api.Dockerfile          # multi-stage, distroless final
│   ├── trainer.Dockerfile
│   └── docker-compose.yml      # api, redis, mlflow, minio, postgres, prom, grafana
├── deploy/
│   ├── k8s/                    # raw manifests
│   └── helm/flowstate/         # Helm chart
├── src/flowstate/
│   ├── __init__.py
│   ├── api/
│   │   ├── main.py             # FastAPI app factory
│   │   ├── routes/
│   │   │   ├── predict.py
│   │   │   ├── health.py
│   │   │   └── admin.py        # model reload, cache flush (authn'd)
│   │   ├── deps.py             # DI: model, cache, settings
│   │   ├── schemas.py          # Pydantic v2
│   │   ├── middleware.py       # request id, timing, auth, rate limit
│   │   └── errors.py
│   ├── inference/
│   │   ├── runtime.py          # ONNX Runtime session, FP16, batching
│   │   ├── batcher.py          # dynamic micro-batching (async queue)
│   │   ├── tokenizer.py
│   │   └── postprocess.py
│   ├── cache/
│   │   ├── redis_cache.py      # get/set, single-flight lock
│   │   └── keys.py             # stable hash of (model_version, input)
│   ├── training/
│   │   ├── data.py             # dataset load + split + version
│   │   ├── train.py            # HF Trainer + MLflow autolog
│   │   ├── evaluate.py         # metrics, confusion matrix, slice eval
│   │   ├── export_onnx.py      # torch → ONNX → FP16
│   │   └── register.py         # promote to MLflow registry
│   ├── monitoring/
│   │   ├── metrics.py          # Prometheus counters/histograms
│   │   ├── tracing.py          # OTel setup
│   │   └── drift.py            # input + prediction drift (Evidently)
│   ├── config.py               # Pydantic Settings (env-driven)
│   └── logging.py              # structlog JSON
├── tests/
│   ├── unit/
│   ├── integration/            # spins up redis + api via testcontainers
│   ├── contract/               # schemathesis against OpenAPI
│   └── load/
│       ├── locustfile.py
│       └── k6_script.js
├── notebooks/                  # EDA only — never the source of truth
├── scripts/
│   ├── seed_data.py
│   ├── benchmark.py
│   └── smoke.sh
└── docs/
    ├── architecture.md
    ├── runbook.md
    ├── slo.md
    └── api.md
```

---

## 4. Phased Delivery Plan

Each phase ends with a **demonstrable, testable artifact**. No phase is "done" until its exit criteria pass in CI.

### Phase 0 — Foundations
- Initialize repo, `pyproject.toml`, ruff/black/mypy/pytest, pre-commit, Makefile targets (`make lint test run build`).
- GitHub Actions skeleton (`ci.yml`): lint, type-check, unit tests, coverage gate ≥ 85%.
- `docker-compose.yml` with Redis, MLflow, MinIO, Postgres, Prometheus, Grafana.
- **Exit:** `make up` brings the full local stack online; CI green on an empty PR.

### Phase 1 — Model & Training Pipeline
- Pick dataset (recommend **AG News** — 4-class, clean, well-known baselines).
- `training/train.py`: HuggingFace `Trainer`, deterministic seeds, MLflow autologging (params, metrics, artifacts, git SHA).
- `training/evaluate.py`: accuracy, macro-F1, per-class F1, latency micro-bench, confusion matrix.
- `training/export_onnx.py`: export to ONNX, convert to FP16, validate numerical parity (cosine sim ≥ 0.999 on eval set).
- `training/register.py`: log to MLflow registry, transition to `Staging`.
- **Exit:** `make train` produces a registered model; eval report committed under `docs/eval/`.

### Phase 2 — Inference Runtime
- ONNX Runtime session with thread tuning (`intra_op_num_threads`, `inter_op`).
- Tokenizer cached in memory, padding strategy fixed.
- **Dynamic batcher:** asyncio queue, `max_batch_size=32`, `max_wait_ms=5`. This is the single biggest throughput lever.
- Warmup on startup (run N dummy inferences before readiness probe flips green).
- **Exit:** `scripts/benchmark.py` shows ≥ 3× throughput vs. naive PyTorch baseline; FP16 vs FP32 latency table in `docs/`.

### Phase 3 — API Layer
- FastAPI app factory pattern, lifespan events for model load + Redis connect + warmup.
- Routes: `POST /v1/predict`, `POST /v1/predict/batch`, `GET /healthz`, `GET /readyz`, `GET /metrics`, `POST /admin/reload` (auth'd).
- Pydantic v2 schemas with strict validation, max input length, request size limits.
- Middleware: request ID, structured logging, timing, JWT/API-key auth, token-bucket rate limit (per-key).
- Graceful shutdown: drain in-flight requests, close Redis pool, flush traces.
- OpenAPI docs auto-generated, contract-tested with **schemathesis**.
- **Exit:** Integration tests (testcontainers) cover happy path + auth + rate limit + malformed input + 5xx propagation.

### Phase 4 — Caching Layer
- Stable cache key: `sha256(model_version || normalized_input)`.
- Read-through cache; on miss, **single-flight** via Redis `SET NX PX` lock to prevent thundering herd / cache stampede.
- TTL configurable per route; jittered to avoid synchronized expiry.
- Cache metrics: hit rate, miss rate, lock contention, eviction rate.
- **Exit:** Load test shows hit-rate ≥ 70% on Zipfian input distribution; latency p50 drops ≥ 50% with cache warm.

### Phase 5 — Observability
- **Metrics (Prometheus):** request count/latency histograms (per route, per status), batch size histogram, queue depth, model inference time, cache hit ratio, GPU/CPU utilization.
- **Logs (structlog → JSON → Loki):** request_id, user_id, model_version, latency, cache_hit, error.
- **Traces (OTel → Tempo/Jaeger):** spans for tokenize → batch_wait → model → postprocess → cache_write.
- **Grafana dashboards:** "Service Overview", "Model Performance", "Infra".
- **Alerts (Alertmanager):** p99 > 50 ms for 5 min, error rate > 1%, cache hit < 40%, drift score > threshold.
- **Exit:** Dashboards screenshotted into `docs/`; alert rules unit-tested with `promtool`.

### Phase 6 — MLOps & CI/CD
- `ci.yml`: lint, mypy, unit, integration, contract, coverage, **Trivy**, **Bandit**, **pip-audit**, SBOM upload.
- `cd.yml`: on tag push → build multi-arch image → sign with `cosign` → push to GHCR → deploy to staging → run smoke + load test → manual approval → deploy to prod.
- **Model promotion workflow:** `workflow_dispatch` triggers MLflow stage transition Staging → Production after eval gates pass (accuracy ≥ baseline, latency ≤ baseline, no drift regression).
- **Zero-downtime deploys:** Kubernetes Deployment with `maxSurge=1, maxUnavailable=0`, readiness probe gated on model warmup.
- **Rollback:** keep last 3 image tags + last 3 model versions; one-command rollback via `helm rollback` and MLflow stage revert.
- **Exit:** A full PR → merge → tag → prod deploy cycle runs end-to-end with zero manual intervention beyond approval.

### Phase 7 — Load Testing & Performance Validation
- **Baseline:** naive FP32 PyTorch, no batching, no cache → record numbers.
- **Optimized:** FP16 ONNX + dynamic batching + Redis cache → record numbers.
- Locust (open model) and k6 (closed model) both run. Cross-validate.
- Publish `docs/perf-report.md` with: hardware spec, methodology, raw CSVs, plots, and the headline numbers (≥ 500 RPS, ≥ 65% latency reduction). **Numbers must be reproducible from `make benchmark`.**
- **Exit:** Nightly load test in CI fails the build on > 10% regression.

### Phase 8 — Drift & Continuous Evaluation
- **Evidently AI** job (scheduled): pulls last 24h of logged inputs/predictions from object storage, computes data drift + prediction drift vs. training distribution, writes report to MLflow + dashboard.
- **Shadow deployment** support: route X% of traffic to a candidate model, compare distributions, never affect user response.
- **Exit:** Drift report generated nightly; alert fires on synthetic drift injection test.

### Phase 9 — Hardening & Docs
- Threat model (STRIDE), pen-test the API with `zap-baseline`.
- Chaos test: kill Redis mid-load, kill an API pod mid-load — system stays within SLO.
- `docs/runbook.md`: oncall playbook (high latency, high error rate, model rollback, cache poisoning, capacity).
- `docs/slo.md`: SLIs, SLOs, error budgets.
- `README.md`: 60-second quickstart + architecture diagram + perf headline + résumé bullet points.

---

## 5. Non-Functional Requirements (the production bar)

| Area | Requirement |
|---|---|
| Latency | p50 ≤ 15 ms, p99 ≤ 50 ms (cache warm), p99 ≤ 120 ms (cold) |
| Throughput | ≥ 500 RPS sustained on 4 vCPU / 8 GB node |
| Availability | 99.9% (documented SLO + error budget) |
| Security | TLS everywhere, JWT/API-key, rate limit, input validation, signed images, SBOM, no secrets in code |
| Reproducibility | `make train` from clean clone produces a model with metrics within ±0.5% of recorded baseline |
| Observability | Every request traceable end-to-end via `request_id` across logs/metrics/traces |
| Recoverability | Rollback to previous model or image in < 2 minutes |
| Test coverage | ≥ 85% line, 100% on `inference/` and `cache/` |

---

## 6. Risks & Mitigations

| Risk | Mitigation |
|---|---|
| FP16 accuracy regression | Numerical parity test on full eval set; gate promotion on accuracy delta |
| Cache stampede | Single-flight lock + jittered TTL |
| Batcher tail latency | Cap `max_wait_ms`; expose queue depth metric; auto-tune in load tests |
| Model registry as SPOF | MLflow backed by Postgres + MinIO with backups |
| GH Actions runner perf variance for load tests | Self-hosted runner or pinned cloud SKU; record hardware in report |
| Scope creep | Each phase has hard exit criteria; nothing ships without them |

---

## 7. Resume-Ready Outcomes (mapped to original bullets)

- **Scalable ML Serving Infrastructure** → Phases 2–3, proven in Phase 7.
- **Performance Optimization (65% ↓ latency, 500+ RPS)** → Phases 2 + 4, measured in Phase 7, regression-gated nightly.
- **Automated MLOps Pipeline (CI/CD, MLflow, zero-downtime)** → Phases 1 + 6.
- **Plus the things hiring managers actually probe:** observability, SLOs, rollback, drift detection, security scanning, chaos testing — all covered in Phases 5, 8, 9.

---

## 8. Immediate Next Actions

1. Confirm dataset choice (AG News recommended).
2. Confirm deployment target for the public demo (local k3d, a small VPS, or a free-tier cloud).
3. Scaffold Phase 0 (repo, tooling, compose stack, CI skeleton).
4. Kick off Phase 1 training run and lock in the baseline numbers we'll optimize against.
