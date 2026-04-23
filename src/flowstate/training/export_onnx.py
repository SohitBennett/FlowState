"""Export a fine-tuned HF classifier to ONNX, quantize to FP16, and gate on parity.

Pipeline:
  1. Load HF model + tokenizer from `<output_dir>/model/`.
  2. Export to ONNX (FP32) with dynamic `batch` and `sequence` axes.
  3. Convert weights to FP16, keeping input/output dtypes unchanged so that
     consumers pass int64 input_ids and receive fp32 logits.
  4. Run both sessions over a sampled subset of the held-out test set and
     fail if the minimum row-wise cosine similarity of the logits is below
     `min_cosine` (default 0.999) — this is the FP16 quality gate.

Writes:
  <output_dir>/onnx/model_fp32.onnx
  <output_dir>/onnx/model_fp16.onnx
  <output_dir>/onnx/parity.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
import torch
from datasets import load_dataset
from onnxconverter_common.float16 import convert_float_to_float16
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from flowstate.training.data import AG_NEWS_LABELS
from flowstate.training.evaluate import logit_cosine_similarity


def export_to_onnx(model_dir: Path, out_path: Path, max_seq_len: int = 128) -> None:
    tokenizer = AutoTokenizer.from_pretrained(str(model_dir))
    model = AutoModelForSequenceClassification.from_pretrained(str(model_dir))
    model.eval()

    dummy = tokenizer(
        "FlowState ONNX export probe.",
        return_tensors="pt",
        padding="max_length",
        truncation=True,
        max_length=max_seq_len,
    )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(
        model,
        args=(dummy["input_ids"], dummy["attention_mask"]),
        f=str(out_path),
        input_names=["input_ids", "attention_mask"],
        output_names=["logits"],
        dynamic_axes={
            "input_ids": {0: "batch", 1: "sequence"},
            "attention_mask": {0: "batch", 1: "sequence"},
            "logits": {0: "batch"},
        },
        opset_version=17,
        do_constant_folding=True,
    )
    onnx.checker.check_model(str(out_path))


def to_fp16(fp32_path: Path, fp16_path: Path) -> None:
    model = onnx.load(str(fp32_path))
    # keep_io_types=True leaves the float inputs/outputs as fp32 so callers
    # don't need to downcast; int64 tensors (input_ids, attention_mask) are
    # untouched because the conversion only targets float tensors.
    fp16_model = convert_float_to_float16(model, keep_io_types=True)
    onnx.save(fp16_model, str(fp16_path))
    onnx.checker.check_model(str(fp16_path))


def _sample_texts(n: int = 256, seed: int = 42) -> list[str]:
    ds = load_dataset("ag_news", split="test")
    ds = ds.shuffle(seed=seed).select(range(min(n, len(ds))))
    return list(ds["text"])


def parity_check(
    fp32_path: Path,
    fp16_path: Path,
    tokenizer_dir: Path,
    samples: list[str] | None = None,
    max_seq_len: int = 128,
    min_cosine: float = 0.999,
) -> dict[str, float]:
    tokenizer = AutoTokenizer.from_pretrained(str(tokenizer_dir))
    texts = samples if samples is not None else _sample_texts()

    enc = tokenizer(
        texts,
        return_tensors="np",
        padding=True,
        truncation=True,
        max_length=max_seq_len,
    )
    feeds = {
        "input_ids": enc["input_ids"].astype(np.int64),
        "attention_mask": enc["attention_mask"].astype(np.int64),
    }

    providers = ["CPUExecutionProvider"]
    sess_fp32 = ort.InferenceSession(str(fp32_path), providers=providers)
    sess_fp16 = ort.InferenceSession(str(fp16_path), providers=providers)

    logits_fp32 = sess_fp32.run(["logits"], feeds)[0]
    logits_fp16 = sess_fp16.run(["logits"], feeds)[0]

    cos = logit_cosine_similarity(logits_fp32, logits_fp16)
    argmax_match = float((logits_fp32.argmax(-1) == logits_fp16.argmax(-1)).mean())

    result = {
        "num_samples": int(len(texts)),
        "min_cosine": float(cos.min()),
        "mean_cosine": float(cos.mean()),
        "argmax_match_rate": argmax_match,
        "min_cosine_threshold": float(min_cosine),
        "passed": bool(cos.min() >= min_cosine),
        "num_classes": len(AG_NEWS_LABELS),
    }
    if not result["passed"]:
        raise ValueError(
            f"FP16 parity failed: min_cosine={result['min_cosine']:.6f} < {min_cosine}"
        )
    return result


def run(output_dir: Path, max_seq_len: int = 128, min_cosine: float = 0.999) -> Path:
    model_dir = output_dir / "model"
    onnx_dir = output_dir / "onnx"
    onnx_dir.mkdir(parents=True, exist_ok=True)

    fp32 = onnx_dir / "model_fp32.onnx"
    fp16 = onnx_dir / "model_fp16.onnx"

    export_to_onnx(model_dir, fp32, max_seq_len=max_seq_len)
    to_fp16(fp32, fp16)
    parity = parity_check(
        fp32, fp16, tokenizer_dir=model_dir, max_seq_len=max_seq_len, min_cosine=min_cosine
    )

    (onnx_dir / "parity.json").write_text(json.dumps(parity, indent=2))
    return onnx_dir


def main() -> None:
    p = argparse.ArgumentParser(description="Export HF model to ONNX FP16 with parity check.")
    p.add_argument("--output-dir", default="artifacts/latest")
    p.add_argument("--max-seq-len", type=int, default=128)
    p.add_argument("--min-cosine", type=float, default=0.999)
    ns = p.parse_args()
    out = run(Path(ns.output_dir), max_seq_len=ns.max_seq_len, min_cosine=ns.min_cosine)
    print(f"[flowstate.export_onnx] wrote {out}")


if __name__ == "__main__":
    main()
