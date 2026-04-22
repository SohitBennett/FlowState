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

## Project layout

See [PLAN.md §3](PLAN.md#3-repository-layout).

## Status

- [x] **Phase 0** — Foundations (tooling, compose stack, CI skeleton, FastAPI skeleton)
- [ ] Phase 1 — Training pipeline (AG News + DistilBERT + MLflow + ONNX FP16)
- [ ] Phase 2 — Inference runtime (ONNX, dynamic batcher, warmup)
- [ ] Phase 3 — API layer (auth, rate limit, contract tests)
- [ ] Phase 4 — Redis caching (single-flight, jittered TTL)
- [ ] Phase 5 — Observability (metrics, logs, traces, dashboards, alerts)
- [ ] Phase 6 — CI/CD + zero-downtime deploys
- [ ] Phase 7 — Load testing & perf validation
- [ ] Phase 8 — Drift & continuous evaluation
- [ ] Phase 9 — Hardening + public VPS deploy
