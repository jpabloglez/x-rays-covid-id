"""In-lung attribution: the metric, and the ways it can mislead."""

from __future__ import annotations

import numpy as np
import pytest
from cxr.masks import MaskError, coverage_summary, in_lung_fraction, load_mask, masked_out
from PIL import Image


def _lung_mask(side=64, coverage_cols=(16, 48)):
    mask = np.zeros((side, side), dtype=bool)
    mask[8 : side - 8, coverage_cols[0] : coverage_cols[1]] = True
    return mask


def test_attribution_entirely_inside_the_lungs_scores_one():
    mask = _lung_mask()
    saliency = np.zeros((64, 64))
    saliency[mask] = 1.0
    assert in_lung_fraction(saliency, mask).in_lung_fraction == pytest.approx(1.0)


def test_attribution_entirely_outside_the_lungs_scores_zero():
    """The shortcut signature: all the mass on borders and corners."""
    mask = _lung_mask()
    saliency = np.zeros((64, 64))
    saliency[~mask] = 1.0
    assert in_lung_fraction(saliency, mask).in_lung_fraction == pytest.approx(0.0)


def test_uniform_saliency_scores_exactly_the_mask_coverage():
    """The trap the lift figure exists to expose.

    Uniform noise scores the mask's own area fraction. Any in-lung number
    reported without its coverage is close to meaningless.
    """
    mask = _lung_mask()
    score = in_lung_fraction(np.ones((64, 64)), mask)
    assert score.in_lung_fraction == pytest.approx(score.lung_coverage)
    assert score.lift == pytest.approx(1.0)
    assert not score.better_than_uniform


def test_lift_separates_a_real_result_from_a_flattering_one():
    mask = _lung_mask()
    focused = np.zeros((64, 64))
    focused[mask] = 1.0

    good = in_lung_fraction(focused, mask)
    uniform = in_lung_fraction(np.ones((64, 64)), mask)

    assert good.lift > uniform.lift
    assert good.better_than_uniform


def test_scale_of_the_saliency_map_does_not_matter():
    mask = _lung_mask()
    rng = np.random.default_rng(0)
    saliency = rng.random((64, 64))
    assert in_lung_fraction(saliency, mask).in_lung_fraction == pytest.approx(
        in_lung_fraction(saliency * 1000, mask).in_lung_fraction
    )


def test_a_coarse_gradcam_map_is_scored_against_a_downsampled_mask():
    """Grad-CAM comes out at the last conv layer's resolution, often 10x10.

    Upsampling it to the mask would invent spatial precision the model never
    had, so the mask comes down to the map instead.
    """
    mask = _lung_mask(side=320, coverage_cols=(80, 240))
    coarse = np.ones((10, 10))
    score = in_lung_fraction(coarse, mask)
    assert 0.0 < score.in_lung_fraction < 1.0


def test_negative_saliency_is_rejected_rather_than_silently_clipped():
    """A signed map has already lost the meaning of 'share of mass'; the
    caller should decide how to handle it, visibly."""
    with pytest.raises(MaskError, match="negative values"):
        in_lung_fraction(np.full((8, 8), -1.0), _lung_mask(8, (2, 6)))


def test_empty_mask_is_rejected():
    with pytest.raises(MaskError, match="mask is empty"):
        in_lung_fraction(np.ones((8, 8)), np.zeros((8, 8), dtype=bool))


def test_full_frame_mask_is_rejected_as_uninformative():
    with pytest.raises(MaskError, match="entire frame"):
        in_lung_fraction(np.ones((8, 8)), np.ones((8, 8), dtype=bool))


def test_zero_saliency_is_rejected():
    with pytest.raises(MaskError, match="sums to zero"):
        in_lung_fraction(np.zeros((64, 64)), _lung_mask())


def test_non_2d_inputs_are_rejected():
    with pytest.raises(MaskError, match="saliency must be 2-D"):
        in_lung_fraction(np.ones((3, 8, 8)), _lung_mask(8, (2, 6)))


def test_load_mask_binarises_and_can_resample(tmp_path):
    path = tmp_path / "mask.png"
    array = np.zeros((32, 32), dtype=np.uint8)
    array[8:24, 8:24] = 255
    Image.fromarray(array, mode="L").save(path)

    assert load_mask(path).dtype == bool
    assert load_mask(path, shape=(16, 16)).shape == (16, 16)


def test_mask_resampling_is_nearest_neighbour(tmp_path):
    """Bilinear would invent half-lung pixels at the boundary that then get
    thresholded arbitrarily."""
    path = tmp_path / "mask.png"
    array = np.zeros((64, 64), dtype=np.uint8)
    array[:, :32] = 255
    Image.fromarray(array, mode="L").save(path)

    resampled = load_mask(path, shape=(8, 8))
    assert set(np.unique(resampled)) <= {True, False}
    assert resampled[:, :4].all()
    assert not resampled[:, 4:].any()


def test_masked_out_blanks_everything_outside_the_lungs():
    """The ablation: re-evaluate on these and accuracy should collapse."""
    mask = _lung_mask()
    image = np.ones((64, 64), dtype=np.float32)
    ablated = masked_out(image, mask)
    assert ablated[mask].sum() == mask.sum()
    assert ablated[~mask].sum() == 0.0


def test_masked_out_resamples_a_mismatched_mask():
    mask = _lung_mask(side=64)
    image = np.ones((32, 32), dtype=np.float32)
    assert masked_out(image, mask).shape == (32, 32)


def test_coverage_summary_reports_the_spread():
    """If coverage varies by source, so does the null baseline, and raw
    fractions from different sources are not comparable."""
    masks = [_lung_mask(coverage_cols=(16, 48)), _lung_mask(coverage_cols=(24, 40))]
    summary = coverage_summary(masks)
    assert summary["min"] < summary["max"]
    assert summary["std"] > 0


def test_coverage_summary_of_nothing_is_empty():
    assert coverage_summary([]) == {}
