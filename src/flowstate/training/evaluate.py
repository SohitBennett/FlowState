"""Pure metric functions used by training, ONNX parity checks, and reports.

Everything in this module is deterministic and free of I/O so it can be
unit-tested without model weights, datasets, or a tracking server.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)


def compute_metrics(eval_pred: Any) -> dict[str, float]:
    """HuggingFace Trainer-compatible metric callback.

    Accepts either a tuple `(logits, labels)` or a `transformers.EvalPrediction`
    (which is iterable and yields the same two arrays).
    """
    logits, labels = eval_pred if isinstance(eval_pred, tuple) else (
        eval_pred.predictions,
        eval_pred.label_ids,
    )
    preds = np.argmax(logits, axis=-1)
    return {
        "accuracy": float(accuracy_score(labels, preds)),
        "macro_f1": float(f1_score(labels, preds, average="macro")),
    }


def full_report(
    labels: np.ndarray, preds: np.ndarray, class_names: list[str] | tuple[str, ...]
) -> dict[str, Any]:
    """Full evaluation payload: accuracy, macro F1, per-class F1, confusion matrix."""
    per_class = f1_score(labels, preds, average=None)
    return {
        "accuracy": float(accuracy_score(labels, preds)),
        "macro_f1": float(f1_score(labels, preds, average="macro")),
        "per_class_f1": {
            name: float(score) for name, score in zip(class_names, per_class, strict=True)
        },
        "confusion_matrix": confusion_matrix(labels, preds).tolist(),
        "classification_report": classification_report(
            labels, preds, target_names=list(class_names), output_dict=True, zero_division=0
        ),
    }


def logit_cosine_similarity(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Row-wise cosine similarity between two (N, C) logit matrices.

    Used to gate FP16 quantization: the minimum row-wise cosine similarity
    between FP32 and FP16 logits must stay above a threshold to accept the
    quantized model.
    """
    if a.shape != b.shape:
        raise ValueError(f"shape mismatch: {a.shape} vs {b.shape}")
    a = a.astype(np.float64)
    b = b.astype(np.float64)
    norm = np.linalg.norm(a, axis=-1) * np.linalg.norm(b, axis=-1)
    # avoid division by zero for all-zero logit rows
    return (a * b).sum(axis=-1) / np.where(norm == 0, 1e-12, norm)
