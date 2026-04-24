"""Pure-math tests for softmax and top_prediction."""

from __future__ import annotations

import numpy as np
import pytest

from flowstate.inference.postprocess import (
    ClassificationResult,
    softmax,
    top_prediction,
)


def test_softmax_sums_to_one_per_row() -> None:
    logits = np.array([[1.0, 2.0, 3.0], [5.0, 5.0, 5.0]])
    probs = softmax(logits)
    np.testing.assert_allclose(probs.sum(axis=-1), [1.0, 1.0], atol=1e-12)


def test_softmax_is_numerically_stable_for_large_logits() -> None:
    logits = np.array([[1000.0, 1001.0, 999.0]])
    probs = softmax(logits)
    assert np.isfinite(probs).all()
    np.testing.assert_allclose(probs.sum(), 1.0, atol=1e-12)


def test_softmax_rejects_non_2d_input() -> None:
    with pytest.raises(ValueError, match="expected 2D logits"):
        softmax(np.array([1.0, 2.0, 3.0]))


def test_top_prediction_picks_argmax_and_score() -> None:
    logits = np.array([[0.1, 5.0, 0.2, 0.3], [9.0, 0.0, 0.0, 0.0]])
    results = top_prediction(logits, ["a", "b", "c", "d"])
    assert [r.label for r in results] == ["b", "a"]
    assert all(0.0 < r.score <= 1.0 for r in results)
    assert results[0].score > 0.9


def test_top_prediction_accepts_tuple_labels() -> None:
    logits = np.array([[0.0, 1.0]])
    results = top_prediction(logits, ("no", "yes"))
    assert results == [ClassificationResult(label="yes", score=results[0].score)]


def test_top_prediction_rejects_label_shape_mismatch() -> None:
    with pytest.raises(ValueError, match="does not match label count"):
        top_prediction(np.zeros((2, 3)), ["a", "b"])
