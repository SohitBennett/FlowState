"""Phase 1 training entry point.

Fine-tunes a transformer classifier on AG News, logs everything to MLflow
(params, metrics, system info, git SHA, dataset fingerprint, artifacts),
and writes the selected checkpoint plus an evaluation report to
`<output_dir>/model/` and `<output_dir>/eval_report.json`.

This module is intentionally I/O-heavy and not unit-tested; correctness is
proven by the integration run (`make train`) and the evaluation metrics in
the resulting report.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
from dataclasses import asdict, dataclass
from pathlib import Path

import mlflow
import numpy as np
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
    set_seed,
)

from flowstate.training.data import AG_NEWS_LABELS, DataConfig, load_and_tokenize
from flowstate.training.evaluate import compute_metrics, full_report


@dataclass
class TrainConfig:
    model_id: str = "distilbert-base-uncased"
    output_dir: str = "artifacts/latest"
    epochs: int = 2
    batch_size: int = 32
    learning_rate: float = 5e-5
    weight_decay: float = 0.01
    warmup_ratio: float = 0.1
    max_seq_len: int = 128
    seed: int = 42
    fp16: bool = False
    bf16: bool = False
    max_train_samples: int | None = None
    experiment_name: str = "flowstate-ag-news"
    run_name: str | None = None


def _git_sha() -> str:
    try:
        return subprocess.check_output(  # noqa: S603
            ["git", "rev-parse", "HEAD"],  # noqa: S607
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown"


def _parse_args() -> TrainConfig:
    p = argparse.ArgumentParser(description="FlowState — fine-tune a text classifier on AG News.")
    p.add_argument("--model-id", default=TrainConfig.model_id)
    p.add_argument("--output-dir", default=TrainConfig.output_dir)
    p.add_argument("--epochs", type=int, default=TrainConfig.epochs)
    p.add_argument("--batch-size", type=int, default=TrainConfig.batch_size)
    p.add_argument("--learning-rate", type=float, default=TrainConfig.learning_rate)
    p.add_argument("--weight-decay", type=float, default=TrainConfig.weight_decay)
    p.add_argument("--warmup-ratio", type=float, default=TrainConfig.warmup_ratio)
    p.add_argument("--max-seq-len", type=int, default=TrainConfig.max_seq_len)
    p.add_argument("--seed", type=int, default=TrainConfig.seed)
    p.add_argument("--fp16", action="store_true", help="mixed-precision training (CUDA)")
    p.add_argument("--bf16", action="store_true", help="bfloat16 training (Ampere+ / CPU)")
    p.add_argument(
        "--max-train-samples",
        type=int,
        default=None,
        help="cap training examples for smoke runs",
    )
    p.add_argument("--experiment-name", default=TrainConfig.experiment_name)
    p.add_argument("--run-name", default=None)
    ns = p.parse_args()
    return TrainConfig(
        model_id=ns.model_id,
        output_dir=ns.output_dir,
        epochs=ns.epochs,
        batch_size=ns.batch_size,
        learning_rate=ns.learning_rate,
        weight_decay=ns.weight_decay,
        warmup_ratio=ns.warmup_ratio,
        max_seq_len=ns.max_seq_len,
        seed=ns.seed,
        fp16=ns.fp16,
        bf16=ns.bf16,
        max_train_samples=ns.max_train_samples,
        experiment_name=ns.experiment_name,
        run_name=ns.run_name,
    )


def run(cfg: TrainConfig) -> Path:
    set_seed(cfg.seed)

    tracking_uri = os.environ.get("MLFLOW_TRACKING_URI", "file:./mlruns")
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(cfg.experiment_name)

    tokenizer = AutoTokenizer.from_pretrained(cfg.model_id)

    label2id = {name: i for i, name in enumerate(AG_NEWS_LABELS)}
    id2label = {i: name for i, name in enumerate(AG_NEWS_LABELS)}
    model = AutoModelForSequenceClassification.from_pretrained(
        cfg.model_id,
        num_labels=len(AG_NEWS_LABELS),
        label2id=label2id,
        id2label=id2label,
    )

    data_cfg = DataConfig(max_seq_len=cfg.max_seq_len, seed=cfg.seed)
    datasets = load_and_tokenize(tokenizer, data_cfg)

    train_ds = datasets["train"]
    if cfg.max_train_samples is not None:
        train_ds = train_ds.select(range(min(cfg.max_train_samples, len(train_ds))))

    output_dir = Path(cfg.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    training_args = TrainingArguments(
        output_dir=str(output_dir / "checkpoints"),
        num_train_epochs=cfg.epochs,
        per_device_train_batch_size=cfg.batch_size,
        per_device_eval_batch_size=cfg.batch_size * 2,
        learning_rate=cfg.learning_rate,
        weight_decay=cfg.weight_decay,
        warmup_ratio=cfg.warmup_ratio,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="macro_f1",
        greater_is_better=True,
        fp16=cfg.fp16,
        bf16=cfg.bf16,
        seed=cfg.seed,
        data_seed=cfg.seed,
        logging_steps=50,
        report_to=["mlflow"],
        run_name=cfg.run_name,
        save_total_limit=2,
        dataloader_num_workers=2,
        remove_unused_columns=True,
    )

    collator = DataCollatorWithPadding(tokenizer)

    with mlflow.start_run(run_name=cfg.run_name) as active_run:
        mlflow.set_tags(
            {
                "git_sha": _git_sha(),
                "dataset": data_cfg.dataset_name,
                "framework": "transformers",
                "model_id": cfg.model_id,
                "train_fingerprint": train_ds._fingerprint,
            }
        )
        mlflow.log_params(
            {f"cfg.{k}": v for k, v in asdict(cfg).items() if v is not None}
        )

        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_ds,
            eval_dataset=datasets["validation"],
            tokenizer=tokenizer,
            data_collator=collator,
            compute_metrics=compute_metrics,
        )

        trainer.train()

        test_pred = trainer.predict(datasets["test"])
        preds = np.argmax(test_pred.predictions, axis=-1)
        report = full_report(test_pred.label_ids, preds, list(AG_NEWS_LABELS))

        report_path = output_dir / "eval_report.json"
        report_path.write_text(json.dumps(report, indent=2))

        mlflow.log_metrics(
            {
                "test_accuracy": report["accuracy"],
                "test_macro_f1": report["macro_f1"],
                **{
                    f"test_f1_{cls}": v for cls, v in report["per_class_f1"].items()
                },
            }
        )
        mlflow.log_artifact(str(report_path))

        model_dir = output_dir / "model"
        trainer.save_model(str(model_dir))
        tokenizer.save_pretrained(str(model_dir))
        mlflow.log_artifacts(str(model_dir), artifact_path="hf-model")

        run_info = {
            "mlflow_run_id": active_run.info.run_id,
            "mlflow_experiment_id": active_run.info.experiment_id,
            "mlflow_tracking_uri": tracking_uri,
            "git_sha": _git_sha(),
            "output_dir": str(output_dir.resolve()),
            "metrics": {
                "test_accuracy": report["accuracy"],
                "test_macro_f1": report["macro_f1"],
            },
        }
        (output_dir / "run_info.json").write_text(json.dumps(run_info, indent=2))

    return output_dir


def main() -> None:
    cfg = _parse_args()
    out = run(cfg)
    print(f"[flowstate.train] wrote {out}")


if __name__ == "__main__":
    main()
