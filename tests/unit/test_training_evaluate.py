"""Pure-math tests for the evaluation utilities."""

from __future__ import annotations

import numpy as np
import pytest

from flowstate.training.evaluate import (
    compute_metrics,
    full_report,
    logit_cosine_similarity,
)


def test_compute_metrics_perfect_predictions_from_tuple() -> None:
    logits = np.array([[0.1, 0.9], [0.8, 0.2], [0.2, 0.8], [0.9, 0.1]])
    labels = np.array([1, 0, 1, 0])
    m = compute_metrics((logits, labels))
    assert m["accuracy"] == 1.0
    assert m["macro_f1"] == 1.0


def test_compute_metrics_accepts_eval_prediction_like() -> None:
    class _EP:
        predictions = np.array([[0.9, 0.1], [0.1, 0.9]])
        label_ids = np.array([0, 1])

    m = compute_metrics(_EP())
    assert m["accuracy"] == 1.0


def test_compute_metrics_partial() -> None:
    logits = np.array([[0.9, 0.1], [0.9, 0.1], [0.1, 0.9], [0.9, 0.1]])
    labels = np.array([0, 1, 1, 0])
    m = compute_metrics((logits, labels))
    assert m["accuracy"] == 0.75
    assert 0.0 < m["macro_f1"] < 1.0


def test_full_report_shape_and_values() -> None:
    labels = np.array([0, 1, 2, 3, 0, 1, 2, 3])
    preds = np.array([0, 1, 2, 3, 0, 1, 2, 3])
    classes = ("World", "Sports", "Business", "Sci/Tech")
    report = full_report(labels, preds, classes)
    assert report["accuracy"] == 1.0
    assert report["macro_f1"] == 1.0
    assert set(report["per_class_f1"]) == set(classes)
    assert all(v == 1.0 for v in report["per_class_f1"].values())
    assert np.array(report["confusion_matrix"]).shape == (4, 4)
    assert "classification_report" in report


def test_full_report_handles_class_imbalance() -> None:
    labels = np.array([0, 0, 0, 1])
    preds = np.array([0, 0, 0, 0])
    report = full_report(labels, preds, ("a", "b"))
    assert report["accuracy"] == 0.75
    assert report["per_class_f1"]["b"] == 0.0


def test_logit_cosine_similarity_identical_vectors() -> None:
    a = np.array([[1.0, 2.0, 3.0], [0.1, 0.2, 0.3]])
    sim = logit_cosine_similarity(a, a)
    np.testing.assert_allclose(sim, [1.0, 1.0], atol=1e-12)


def test_logit_cosine_similarity_opposite_vectors() -> None:
    a = np.array([[1.0, 2.0, 3.0]])
    sim = logit_cosine_similarity(a, -a)
    np.testing.assert_allclose(sim, [-1.0], atol=1e-12)


def test_logit_cosine_similarity_handles_zero_vector() -> None:
    a = np.array([[0.0, 0.0, 0.0]])
    b = np.array([[1.0, 2.0, 3.0]])
    sim = logit_cosine_similarity(a, b)
    assert np.isfinite(sim).all()


def test_logit_cosine_similarity_near_identity_small_perturbation() -> None:
    rng = np.random.default_rng(0)
    a = rng.normal(size=(32, 4))
    b = a + 1e-4 * rng.normal(size=(32, 4))
    sim = logit_cosine_similarity(a, b)
    assert sim.min() > 0.999


def test_logit_cosine_similarity_shape_mismatch_raises() -> None:
    with pytest.raises(ValueError, match="shape mismatch"):
        logit_cosine_similarity(np.zeros((2, 3)), np.zeros((2, 4)))
