"""Metrics, and the ways they can quietly lie."""

from __future__ import annotations

import numpy as np
import pytest
from cxr.data import CLASSES
from cxr.evaluate import evaluate, expected_calibration_error, softmax


def _confident(labels, correct_fraction=1.0, seed=0):
    """Probabilities that put mass on the true class for most rows."""
    rng = np.random.default_rng(seed)
    probabilities = np.full((len(labels), len(CLASSES)), 0.05)
    for position, label in enumerate(labels):
        target = label if rng.random() < correct_fraction else (label + 1) % len(CLASSES)
        probabilities[position, target] = 0.9
    return probabilities / probabilities.sum(axis=1, keepdims=True)


def test_a_perfect_model_scores_one():
    labels = np.array([0, 1, 2] * 10)
    scored = evaluate(labels, _confident(labels))
    assert scored.macro_auc == pytest.approx(1.0)
    assert scored.accuracy == pytest.approx(1.0)


def test_the_majority_class_predictor_is_caught_by_balanced_accuracy():
    """On a corpus that is 49% normal, accuracy alone rewards learning nothing."""
    labels = np.array([0] * 80 + [1] * 12 + [2] * 8)
    always_normal = np.tile([0.9, 0.05, 0.05], (len(labels), 1))
    scored = evaluate(labels, always_normal)

    assert scored.accuracy == pytest.approx(0.8)
    assert scored.balanced_accuracy == pytest.approx(1 / 3)


def test_a_class_with_no_positives_is_nan_not_half():
    """Reporting 0.5 would put an invented number into the macro average."""
    labels = np.array([0, 0, 1, 1])
    scored = evaluate(labels, _confident(labels))
    assert np.isnan(scored.per_class_auc[CLASSES[2]])
    assert not np.isnan(scored.macro_auc)


def test_confusion_rows_are_true_classes():
    labels = np.array([0, 0, 1, 2])
    predictions = np.array([[0.9, 0.05, 0.05]] * 4)
    scored = evaluate(labels, predictions)
    assert scored.confusion[0] == [2, 0, 0]
    assert scored.confusion[1] == [1, 0, 0]


def test_shape_mismatch_is_rejected():
    with pytest.raises(ValueError, match="probabilities must be"):
        evaluate(np.array([0, 1]), np.zeros((2, 5)))


def test_softmax_temperature_softens_without_reordering():
    """Calibration must not change which class wins, only how sure it sounds."""
    logits = np.array([[3.0, 1.0, 0.0], [0.5, 2.5, 1.0]])
    sharp = softmax(logits)
    soft = softmax(logits, temperature=3.0)

    assert (sharp.argmax(axis=1) == soft.argmax(axis=1)).all()
    assert soft.max(axis=1).max() < sharp.max(axis=1).max()


def test_calibration_error_is_zero_when_confidence_matches_accuracy():
    labels = np.array([0] * 100)
    perfect = np.tile([1.0, 0.0, 0.0], (100, 1))
    assert expected_calibration_error(labels, perfect) == pytest.approx(0.0, abs=1e-9)


def test_calibration_error_catches_confident_wrongness():
    """The failure AUC cannot see: right ranking, badly wrong probabilities."""
    labels = np.array([1] * 100)
    overconfident = np.tile([0.99, 0.005, 0.005], (100, 1))
    assert expected_calibration_error(labels, overconfident) > 0.9


def test_fit_temperature_cools_an_overconfident_model():
    torch = pytest.importorskip("torch")
    assert torch is not None

    from cxr.evaluate import fit_temperature

    rng = np.random.default_rng(0)
    labels = rng.integers(0, 3, size=400)
    # Logits far larger than the evidence warrants: right most of the time,
    # certain all of the time. Temperature should come out above 1.
    logits = np.full((400, 3), -6.0)
    for position, label in enumerate(labels):
        logits[position, label if rng.random() < 0.75 else (label + 1) % 3] = 6.0

    assert fit_temperature(logits, labels) > 1.0
