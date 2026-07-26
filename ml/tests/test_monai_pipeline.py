"""The training pipeline, and the measurement that shaped it."""

from __future__ import annotations

import numpy as np
import pytest
from cxr.preprocessing import PreprocessingSpec
from cxr.preprocessing import reference as ref
from PIL import Image

pytest.importorskip("monai")


def _chest(shape=(240, 200), seed=0):
    rng = np.random.default_rng(seed)
    return (rng.random(shape) * 200 + 20).astype(np.float32)


def test_monai_and_pillow_bilinear_disagree_when_downsampling():
    """The measurement that killed the two-executor design.

    Radiographs always downsample -- a 2500px chest becomes 320px -- and here
    MONAI's Resize and Pillow's differ by far more than the normalisation
    scale. A model trained through one and served through the other is being
    fed measurably different images.

    This test documents a property of the libraries, not a bug in this code.
    If it ever fails because the gap closed, the two-executor design becomes
    viable again and this module should be revisited.
    """
    from monai.transforms import Resize

    image = np.random.default_rng(0).random((240, 240)).astype(np.float32)
    pillow = np.asarray(
        Image.fromarray(image, mode="F").resize((64, 64), Image.BILINEAR), dtype=np.float32
    )
    monai_out = np.asarray(
        Resize(spatial_size=(64, 64), mode="bilinear", align_corners=False)(image[None])[0],
        dtype=np.float32,
    )

    assert np.abs(monai_out - pillow).max() > 0.1


def test_the_two_agree_when_upsampling():
    """Confirms the cause is filter support on downsample, not a bug."""
    from monai.transforms import Resize

    image = np.random.default_rng(1).random((32, 32)).astype(np.float32)
    pillow = np.asarray(
        Image.fromarray(image, mode="F").resize((64, 64), Image.BILINEAR), dtype=np.float32
    )
    monai_out = np.asarray(
        Resize(spatial_size=(64, 64), mode="bilinear", align_corners=False)(image[None])[0],
        dtype=np.float32,
    )

    assert np.abs(monai_out - pillow).max() < 1e-5


def test_inference_pipeline_is_exactly_the_reference():
    """The whole point: one implementation of deterministic preprocessing."""
    from cxr.preprocessing.monai_pipeline import inference_pipeline

    spec = PreprocessingSpec(target_size=64)
    image = _chest()
    assert np.array_equal(inference_pipeline(spec)(image), ref.apply(image, spec))


def test_inference_pipeline_accepts_a_channel_first_input():
    from cxr.preprocessing.monai_pipeline import inference_pipeline

    spec = PreprocessingSpec(target_size=32)
    image = _chest((64, 64))
    assert np.array_equal(inference_pipeline(spec)(image[None]), ref.apply(image, spec))


def test_inference_pipeline_is_deterministic():
    from cxr.preprocessing.monai_pipeline import inference_pipeline

    pipeline = inference_pipeline(PreprocessingSpec(target_size=48))
    image = _chest()
    assert np.array_equal(pipeline(image), pipeline(image))


def test_training_pipeline_produces_the_declared_shape():
    from cxr.preprocessing.monai_pipeline import training_pipeline

    spec = PreprocessingSpec(target_size=64)
    output = np.asarray(training_pipeline(spec, seed=0)(_chest()))
    assert output.shape == (3, 64, 64)
    assert np.isfinite(output).all()


def test_training_pipeline_actually_augments():
    """A pipeline that returns the input unchanged is not augmenting."""
    from cxr.preprocessing.monai_pipeline import inference_pipeline, training_pipeline

    spec = PreprocessingSpec(target_size=64)
    image = _chest()
    deterministic = inference_pipeline(spec)(image)

    outputs = [np.asarray(training_pipeline(spec, seed=seed)(image)) for seed in range(6)]
    assert any(not np.allclose(output, deterministic) for output in outputs)


def test_augmentation_is_reproducible_from_a_seed():
    from cxr.preprocessing.monai_pipeline import training_pipeline

    spec = PreprocessingSpec(target_size=32)
    image = _chest((64, 64))
    first = np.asarray(training_pipeline(spec, seed=7)(image))
    second = np.asarray(training_pipeline(spec, seed=7)(image))
    assert np.allclose(first, second)


def test_augmentation_does_not_flip_horizontally():
    """Flipping destroys laterality and would mask situs anomalies. This is
    where a generic vision recipe gets radiographs wrong.
    """
    from cxr.preprocessing.monai_pipeline import augmentation

    names = {type(t).__name__ for t in augmentation(PreprocessingSpec()).transforms}
    assert not any("Flip" in name for name in names)


def test_augmentation_includes_dropout_to_break_marker_reliance():
    from cxr.preprocessing.monai_pipeline import augmentation

    names = {type(t).__name__ for t in augmentation(PreprocessingSpec()).transforms}
    assert "RandCoarseDropout" in names
