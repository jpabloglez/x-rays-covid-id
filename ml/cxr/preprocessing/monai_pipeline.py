"""Training-time pipeline: one deterministic implementation, MONAI augmentation.

The original plan was two executors -- MONAI for training, numpy for serving --
with an equivalence test keeping them honest. Measuring them killed that idea,
and the measurement is in test_monai_pipeline.py.

MONAI's bilinear Resize and Pillow's differ by up to 0.46 on a [0, 1] image
when downsampling, 0.10 even with anti-aliasing enabled. Radiographs always
downsample: a 2500px chest becomes 320px. That difference is larger than the
normalisation scale, so a model trained through MONAI and served through
Pillow is being fed measurably different images. Upsampling agrees to float
precision, which confirms the cause is filter support rather than a bug in
either library.

No tolerance would make that test meaningful: passing at 0.10 is not agreement,
and the drift it permits is exactly what the test existed to prevent.

So the deterministic part of preprocessing has exactly ONE implementation --
the numpy reference, which the serving side can reproduce -- and MONAI is used
only for augmentation. That asymmetry is principled rather than a compromise:
augmentation is train-only by definition and has no serving counterpart to
diverge from, whereas deterministic preprocessing must be identical or the
model is being lied to at inference.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from cxr.preprocessing.reference import apply as reference_apply
from cxr.preprocessing.spec import PreprocessingSpec


class DeterministicPreprocess:
    """MONAI-compatible callable wrapping the reference implementation.

    Deliberately not built from monai.transforms. Composing MONAI's own
    Resize here would reintroduce the divergence measured above, silently,
    the moment someone found this class and "simplified" it.
    """

    def __init__(self, spec: PreprocessingSpec) -> None:
        self.spec = spec

    def __call__(self, image: np.ndarray) -> np.ndarray:
        if image.ndim == 3 and image.shape[0] == 1:
            image = image[0]
        return reference_apply(image, self.spec)


def augmentation(spec: PreprocessingSpec) -> Any:
    """Train-time augmentation. Never applied at inference.

    Horizontal flip is absent on purpose: it destroys laterality and would mask
    situs anomalies, which is where a generic vision recipe gets radiographs
    wrong. Corner-weighted dropout is present on purpose, to break reliance on
    laterality markers and burnt-in text.
    """
    from monai.transforms import (
        Compose,
        RandAdjustContrast,
        RandAffine,
        RandCoarseDropout,
        RandGaussianNoise,
    )

    size = spec.target_size
    return Compose(
        [
            RandAffine(
                prob=0.7,
                rotate_range=np.deg2rad(10),
                translate_range=(0.05 * size, 0.05 * size),
                scale_range=(0.05, 0.05),
                padding_mode="zeros",
            ),
            RandGaussianNoise(prob=0.3, std=0.02),
            RandAdjustContrast(prob=0.3, gamma=(0.8, 1.25)),
            RandCoarseDropout(
                holes=2,
                spatial_size=(size // 8, size // 8),
                fill_value=0.0,
                prob=0.3,
            ),
        ]
    )


def training_pipeline(spec: PreprocessingSpec, *, seed: int | None = None) -> Any:
    """Deterministic preprocessing, then augmentation. Order matters.

    Augmenting before normalisation would apply noise and contrast in raw
    intensity units, which mean different things per source -- so the same
    nominal augmentation would be a different perturbation depending on where
    the image came from, and that is a confound wearing a helpful disguise.
    """
    from monai.transforms import Compose

    # Seed the outer Compose, not the inner one. Constructing a Compose
    # re-seeds its randomisable children from the global state, so a seed
    # applied before wrapping is silently discarded -- which looks like
    # working reproducibility right up until you try to reproduce a run.
    pipeline = Compose([DeterministicPreprocess(spec), augmentation(spec)])
    if seed is not None:
        pipeline.set_random_state(seed=seed)
    return pipeline


def inference_pipeline(spec: PreprocessingSpec) -> DeterministicPreprocess:
    """What serving must reproduce. No randomness, no MONAI."""
    return DeterministicPreprocess(spec)
