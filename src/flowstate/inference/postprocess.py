"""Pure post-processing of model logits. No I/O, fully unit-testable."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class ClassificationResult:
    label: str
    score: float


def softmax(logits: np.ndarray) -> np.ndarray:
    """Numerically stable row-wise softmax on a (batch, classes) matrix."""
    if logits.ndim != 2:
        raise ValueError(f"expected 2D logits, got shape {logits.shape}")
    shifted = logits - logits.max(axis=-1, keepdims=True)
    exp = np.exp(shifted)
    return exp / exp.sum(axis=-1, keepdims=True)


def top_prediction(
    logits: np.ndarray, labels: list[str] | tuple[str, ...]
) -> list[ClassificationResult]:
    """Convert (batch, num_labels) logits into argmax label + probability."""
    if logits.shape[-1] != len(labels):
        raise ValueError(
            f"logits last dim {logits.shape[-1]} does not match label count {len(labels)}"
        )
    probs = softmax(logits)
    idx = probs.argmax(axis=-1)
    return [
        ClassificationResult(label=labels[i], score=float(probs[row, i]))
        for row, i in enumerate(idx)
    ]
