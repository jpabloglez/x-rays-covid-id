"""The plausibility gate, which is the only check available on the far side."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from cxr.segment import (
    MAX_AREA_FRACTION,
    MIN_AREA_FRACTION,
    SegmenterReport,
    assess,
    dice,
    summarise_predictions,
)


def _lungs(side=64, area=0.30):
    """Two vertical blobs, roughly where lungs are."""
    mask = np.zeros((side, side), dtype=bool)
    height = int(side * 0.55)
    width = max(1, int(side * area / (2 * 0.55)))
    top = int(side * 0.15)
    mask[top : top + height, side // 4 - width // 2 : side // 4 + width // 2] = True
    mask[top : top + height, 3 * side // 4 - width // 2 : 3 * side // 4 + width // 2] = True
    return mask


def test_a_normal_looking_mask_passes():
    quality = assess(_lungs())
    assert quality.plausible, quality.failures
    assert MIN_AREA_FRACTION < quality.area_fraction < MAX_AREA_FRACTION


def test_an_empty_mask_is_rejected():
    """The commonest way a transferred segmenter fails: it predicts nothing."""
    quality = assess(np.zeros((64, 64), dtype=bool))
    assert not quality.plausible
    assert quality.components == 0
    assert any("area" in reason for reason in quality.failures)


def test_a_mask_covering_the_whole_frame_is_rejected():
    """The second commonest: it predicts everything, and the ablation then
    blanks the entire image while looking like it worked."""
    quality = assess(np.ones((64, 64), dtype=bool))
    assert not quality.plausible
    assert any("area" in reason for reason in quality.failures)


def test_a_single_lung_is_rejected():
    mask = _lungs()
    mask[:, 32:] = False
    quality = assess(mask)
    assert not quality.plausible
    assert any("side" in reason for reason in quality.failures)


def test_a_mask_sitting_at_the_bottom_is_rejected():
    """Lungs are not in the abdomen. A mask that drifts there has locked onto
    something other than the lung fields."""
    mask = np.zeros((64, 64), dtype=bool)
    mask[54:62, 10:25] = True
    mask[54:62, 39:54] = True
    quality = assess(mask)
    assert not quality.plausible
    assert any("centroid" in reason for reason in quality.failures)


def test_components_counts_the_two_lungs():
    assert assess(_lungs()).components == 2


def test_components_is_capped_rather_than_exhaustive():
    """A shattered mask is rejected on area anyway; the exact count is noise."""
    speckle = np.zeros((64, 64), dtype=bool)
    speckle[::4, ::4] = True
    assert assess(speckle).components <= 8


def test_assess_rejects_a_non_2d_mask():
    with pytest.raises(ValueError, match="2D mask"):
        assess(np.zeros((3, 64, 64), dtype=bool))


def test_dice_is_one_for_identical_masks():
    mask = _lungs()
    assert dice(mask, mask) == pytest.approx(1.0)


def test_dice_is_zero_for_disjoint_masks():
    left = np.zeros((32, 32), dtype=bool)
    left[:, :16] = True
    assert dice(left, ~left) == pytest.approx(0.0)


def test_dice_of_two_empty_masks_is_one_not_nan():
    assert dice(np.zeros((8, 8), bool), np.zeros((8, 8), bool)) == 1.0


def test_statistics_are_reported_per_collection():
    """The comparison is the whole point: a target whose masks are
    systematically smaller means the segmenter behaves differently there."""
    predictions = pd.DataFrame(
        [{"source": "rsna_pneumonia", "area_fraction": 0.20,
          "side_imbalance": 0.1, "vertical_centroid": 0.45}]
    )
    reference = pd.DataFrame(
        [{"source": "covid_radiography", "area_fraction": 0.32,
          "side_imbalance": 0.05, "vertical_centroid": 0.44}]
    )
    stats = summarise_predictions(predictions, reference)
    assert set(stats) == {"rsna_pneumonia", "covid_radiography"}
    assert stats["rsna_pneumonia"]["area_fraction"] == pytest.approx(0.20)


def test_the_report_says_the_dice_describes_only_the_source_domain():
    """A Dice measured on the source is not evidence about the target, and the
    report has to say so where the number is printed."""
    text = SegmenterReport(
        val_dice=0.95, val_images=1216, target_images=14863,
        implausible=120, reasons={"area": 120}, statistics={},
    ).render()
    assert "source domain only" in text
    assert "no ground truth on" in text
    assert "14743 of 14863 plausible" in text
