.PHONY: help install dev lint fmt type test cov run up down build train export register pipeline benchmark smoke clean

PYTHON ?= python
PIP ?= pip

help:
	@echo "FlowState — common targets"
	@echo "  install     Install runtime deps"
	@echo "  dev         Install dev + train extras + pre-commit hooks"
	@echo "  lint        Ruff lint"
	@echo "  fmt         Ruff + Black format"
	@echo "  type        mypy strict"
	@echo "  test        Pytest with coverage gate"
	@echo "  run         Run API locally (uvicorn, reload)"
	@echo "  up / down   docker compose up/down (full stack)"
	@echo "  build       Build API container"
	@echo "  train       Fine-tune model on AG News + log to MLflow (Phase 1)"
	@echo "  export      Export trained model to ONNX FP16 with parity gate"
	@echo "  register    Register ONNX model to MLflow + set staging alias"
	@echo "  pipeline    Run train -> export -> register end-to-end"
	@echo "  benchmark   Run perf benchmark (Phase 7)"
	@echo "  smoke       End-to-end smoke test"

install:
	$(PIP) install -e .

dev:
	$(PIP) install -e ".[dev,train]"
	pre-commit install

lint:
	ruff check src tests

fmt:
	ruff check --fix src tests
	black src tests

type:
	mypy

test:
	pytest

run:
	uvicorn flowstate.api.main:app --reload --host 0.0.0.0 --port 8000

up:
	docker compose -f docker/docker-compose.yml up -d --build

down:
	docker compose -f docker/docker-compose.yml down -v

build:
	docker build -f docker/api.Dockerfile -t flowstate/api:dev .

train:
	$(PYTHON) -m flowstate.training.train $(TRAIN_ARGS)

export:
	$(PYTHON) -m flowstate.training.export_onnx $(EXPORT_ARGS)

register:
	$(PYTHON) -m flowstate.training.register $(REGISTER_ARGS)

pipeline: train export register

benchmark:
	$(PYTHON) scripts/benchmark.py

smoke:
	bash scripts/smoke.sh

clean:
	rm -rf .pytest_cache .ruff_cache .mypy_cache build dist *.egg-info .coverage htmlcov
