#!/usr/bin/env bash
# End-to-end smoke test against a running local stack.
set -euo pipefail

BASE_URL="${BASE_URL:-http://localhost:8000}"

echo "→ healthz"
curl -fsS "$BASE_URL/healthz" | tee /dev/stderr | grep -q '"status":"ok"'

echo
echo "→ readyz"
curl -fsS "$BASE_URL/readyz" >/dev/null

echo "→ metrics"
curl -fsS "$BASE_URL/metrics" | grep -q flowstate_requests_total

echo
echo "smoke ok"
