"""The ablation that decides what a Track 1 score is worth."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from cxr.confound import (
    Ablation,
    blank_lungs,
    coverage_note,
    interpret,
    lung_masking_ablation,
    maskable,
)
from cxr.data import CLASSES
from cxr.evaluate import evaluate


def _probabilities(labels, correct_fraction, seed=0):
    """Predictions that are right `correct_fraction` of the time."""
    rng = np.random.default_rng(seed)
    out = np.full((len(labels), len(CLASSES)), 0.1)
    for position, label in enumerate(labels):
        target = label if rng.random() < correct_fraction else rng.integers(0, len(CLASSES))
        out[position, target] += 2.0
    return out / out.sum(axis=1, keepdims=True)


LABELS = np.array([0, 1, 2] * 40)


def _ablation(baseline_fraction, ablated_fraction):
    return lung_masking_ablation(
        LABELS,
        _probabilities(LABELS, baseline_fraction, seed=1),
        LABELS,
        _probabilities(LABELS, ablated_fraction, seed=2),
        images=len(LABELS),
    )


def test_retention_is_zero_when_the_ablation_destroys_the_signal():
    result = _ablation(1.0, 0.0)
    assert result.baseline.macro_auc > 0.95
    assert result.retention < 0.15


def test_retention_is_high_when_masking_changes_nothing():
    """The DeGrave et al. finding: blank the lungs, keep the score."""
    result = _ablation(1.0, 1.0)
    assert result.retention == pytest.approx(1.0, abs=0.05)


def test_comparing_different_rows_is_refused():
    """Otherwise the ablation measures the population change instead."""
    with pytest.raises(ValueError, match="same rows in the same order"):
        lung_masking_ablation(
            LABELS, _probabilities(LABELS, 1.0), LABELS[::-1],
            _probabilities(LABELS, 1.0), images=len(LABELS),
        )


def test_a_chance_baseline_makes_the_ablation_meaningless():
    uninformative = np.tile([1 / 3, 1 / 3, 1 / 3], (len(LABELS), 1))
    result = Ablation(
        name="x", images=len(LABELS),
        baseline=evaluate(LABELS, uninformative), ablated=evaluate(LABELS, uninformative),
    )
    assert np.isnan(result.retention)
    assert "says nothing" in interpret(result)


def test_interpretation_escalates_with_retention():
    assert "must not be reported as a diagnostic result" in interpret(_ablation(1.0, 1.0))
    assert "mostly reading the lung fields" in interpret(_ablation(1.0, 0.0))


def test_retention_means_the_opposite_when_the_lungs_are_what_was_kept():
    """The reporting bug this guards against: the same 99% retention is damning
    for `lungs removed` and reassuring for `lungs only`, and a single hardcoded
    verdict reports one experiment's conclusion under the other's name."""
    shortcut = _ablation(1.0, 1.0)
    kept = Ablation(
        name="Lungs only", images=shortcut.images,
        baseline=shortcut.baseline, ablated=shortcut.ablated,
        removed="everything outside the lungs", retention_is_shortcut=False,
    )
    assert "must not be reported" in interpret(shortcut)
    assert "must not be reported" not in interpret(kept)
    assert "anatomy does carry most" in interpret(kept)


def test_a_low_retention_with_only_the_lungs_kept_is_the_bad_news():
    poor = _ablation(1.0, 0.0)
    kept = Ablation(
        name="Lungs only", images=poor.images, baseline=poor.baseline, ablated=poor.ablated,
        removed="everything outside the lungs", retention_is_shortcut=False,
    )
    assert "cannot do the job from the anatomy alone" in interpret(kept)


def test_the_wording_names_what_remains_not_what_was_removed():
    """Retention describes the signal left behind, so the sentence must name
    what survived the ablation. Naming the removed region instead inverts the
    claim while reading perfectly fluently -- which is how it shipped twice."""
    base = _ablation(1.0, 1.0)
    kept = Ablation(
        name="Lungs only", images=10, baseline=base.baseline, ablated=base.ablated,
        removed="everything outside the lungs", remaining="the lung fields",
        retention_is_shortcut=False,
    )
    text = interpret(kept)
    assert "available from the lung fields alone" in text
    assert "everything outside the lungs" not in text


def test_a_good_result_is_still_not_called_sufficient():
    """A low retention rules out one shortcut, not every shortcut."""
    text = interpret(_ablation(1.0, 0.0))
    assert "not sufficient" in text
    assert "view position" in text


def test_blank_lungs_removes_only_the_masked_region():
    image = np.ones((3, 8, 8), dtype=np.float32)
    mask = np.zeros((8, 8), dtype=np.uint8)
    mask[2:6, 2:6] = 1

    blanked = blank_lungs(image, mask)
    assert blanked[:, 2:6, 2:6].sum() == 0
    # The frame, the border and any burnt-in text survive on purpose: cropping
    # to the lungs would remove those too and confound the two explanations.
    assert blanked.sum() == image.sum() - 3 * 16


def test_blank_lungs_rejects_a_mismatched_mask():
    with pytest.raises(ValueError, match="does not match"):
        blank_lungs(np.ones((3, 8, 8)), np.ones((4, 4)))


def _frame(sources_and_masks):
    return pd.DataFrame(
        [
            {"source": source, "label": label, "mask_path": mask}
            for source, label, mask in sources_and_masks
        ]
    )


def test_maskable_selects_only_rows_with_a_mask():
    frame = _frame([("a", "covid", "m1.png"), ("b", "normal", None)])
    assert len(maskable(frame)) == 1


def test_coverage_note_names_what_is_not_covered():
    """The ablation must never be quoted as if it covered the whole test set."""
    frame = _frame(
        [("covid_radiography", "covid", "m.png")] * 3 + [("rsna_pneumonia", "normal", None)] * 5
    )
    note = coverage_note(frame, maskable(frame))
    assert "3 of 8" in note
    assert "rsna_pneumonia" in note
    assert "one collection (covid_radiography)" in note


def test_coverage_note_stops_claiming_one_collection_once_masks_span_two():
    """Once segmentation extends coverage to a second collection the caveat is
    no longer true, and leaving it in understates the result as badly as the
    reverse would overstate it."""
    frame = _frame(
        [("covid_radiography", "covid", "m.png")] * 3
        + [("rsna_pneumonia", "normal", "m.png")] * 4
        + [("rsna_pneumonia", "normal", None)]
    )
    note = coverage_note(frame, maskable(frame))
    assert "span 2 collections" in note
    assert "single acquisition pipeline" in note
    assert "cannot separate anatomy" not in note


def test_coverage_note_says_so_when_everything_is_covered():
    frame = _frame([("covid_radiography", "covid", "m.png")] * 4)
    assert "Every test image" in coverage_note(frame, maskable(frame))
