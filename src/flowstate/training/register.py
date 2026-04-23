"""Register an exported ONNX model to the MLflow registry and set a stage alias.

Reads `run_info.json` (written by train.py) to locate the MLflow run, attaches
the ONNX artifacts and parity report to that run, registers a new model
version under `model_name`, and atomically moves the `staging` alias to point
at it. Production promotion is a separate gated step (Phase 6 CI workflow).
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import mlflow
from mlflow.tracking import MlflowClient


def register(
    output_dir: Path,
    model_name: str,
    alias: str = "staging",
) -> dict[str, str]:
    run_info_path = output_dir / "run_info.json"
    if not run_info_path.exists():
        raise FileNotFoundError(f"missing {run_info_path}; run train.py first")
    run_info = json.loads(run_info_path.read_text())
    run_id = run_info["mlflow_run_id"]

    tracking_uri = os.environ.get("MLFLOW_TRACKING_URI", run_info.get("mlflow_tracking_uri"))
    mlflow.set_tracking_uri(tracking_uri)
    client = MlflowClient()

    onnx_dir = output_dir / "onnx"
    fp16 = onnx_dir / "model_fp16.onnx"
    parity = onnx_dir / "parity.json"
    if not fp16.exists() or not parity.exists():
        raise FileNotFoundError("ONNX FP16 model or parity report missing; run export_onnx first")

    with mlflow.start_run(run_id=run_id):
        mlflow.log_artifact(str(fp16), artifact_path="onnx")
        mlflow.log_artifact(str(parity), artifact_path="onnx")

    model_uri = f"runs:/{run_id}/onnx"
    mv = mlflow.register_model(model_uri=model_uri, name=model_name)

    client.set_registered_model_alias(model_name, alias, mv.version)

    summary = {
        "model_name": model_name,
        "version": str(mv.version),
        "alias": alias,
        "run_id": run_id,
        "tracking_uri": tracking_uri or "",
    }
    (output_dir / "registry.json").write_text(json.dumps(summary, indent=2))
    return summary


def main() -> None:
    p = argparse.ArgumentParser(description="Register ONNX model to MLflow and set alias.")
    p.add_argument("--output-dir", default="artifacts/latest")
    p.add_argument("--model-name", default="ag-news-distilbert")
    p.add_argument("--alias", default="staging", choices=["staging", "production", "archived"])
    ns = p.parse_args()
    info = register(Path(ns.output_dir), model_name=ns.model_name, alias=ns.alias)
    print(f"[flowstate.register] {json.dumps(info)}")


if __name__ == "__main__":
    main()
