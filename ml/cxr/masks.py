"""Lung masks, and the attribution metric that makes them worth carrying.

In-lung attribution fraction -- the share of a saliency map's mass falling
inside the segmented lung field -- is the number that distinguishes this
project from every other COVID X-ray repository. A model at 0.91 AUROC with
0.85 in-lung is a better result than one at 0.97 with 0.40, and being able to
show that trade-off rather than assert it is the judgment the shortcut-learning
literature found missing across 415 papers.

The metric has one trap, handled below: the raw fraction is meaningless without
the mask's own coverage. If the lung field occupies 60% of the frame, uniform
noise scores 0.60, so an unqualified "0.65 in-lung" is close to worthless.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image


class MaskError(ValueError):
    """Raised when a mask cannot be reconciled with the map it scores."""


@dataclass(frozen=True)
class AttributionScore:
    """Where a saliency map put its mass, relative to the lung field."""

    in_lung_fraction: float
    """Share of saliency mass inside the lungs, in [0, 1]."""

    lung_coverage: float
    """Share of the frame the lung field occupies -- the null baseline."""

    lift: float
    """in_lung_fraction / lung_coverage. 1.0 means no better than uniform.

    This is the number to read. A model scoring 0.85 in-lung on a mask covering
    0.80 of the frame has a lift of 1.06 and is barely looking at lungs at all.
    """

    @property
    def better_than_uniform(self) -> bool:
        return self.lift > 1.0


def load_mask(path: Path, shape: tuple[int, int] | None = None) -> np.ndarray:
    """Read a segmentation mask as a boolean array, optionally resampled.

    Nearest-neighbour on purpose: a mask is a set membership, and bilinear
    resampling would invent half-lung pixels at the boundary that then get
    thresholded arbitrarily.
    """
    with Image.open(path) as image:
        mask = image.convert("L")
        if shape is not None and mask.size != (shape[1], shape[0]):
            mask = mask.resize((shape[1], shape[0]), Image.NEAREST)
        return np.asarray(mask) > 127


def in_lung_fraction(saliency: np.ndarray, mask: np.ndarray) -> AttributionScore:
    """Score a saliency map against a lung mask.

    `saliency` may be any non-negative 2-D map -- Grad-CAM, HiResCAM, occlusion
    sensitivity. It is renormalised here, so scale does not matter. Grad-CAM
    output is typically far coarser than the mask, so the mask is resampled
    down to meet it rather than the map being smoothed up: upsampling a 10x10
    attribution map to 320x320 invents spatial precision the model never had.
    """
    if saliency.ndim != 2:
        raise MaskError(f"saliency must be 2-D, got shape {saliency.shape}")
    if mask.ndim != 2:
        raise MaskError(f"mask must be 2-D, got shape {mask.shape}")

    saliency = np.asarray(saliency, dtype=np.float64)
    if np.any(saliency < 0):
        # A signed map has already lost the meaning of "share of mass". Relu
        # it deliberately at the call site instead, so the choice is visible.
        raise MaskError(
            "saliency contains negative values; clip or relu it first so the "
            "fraction remains interpretable as a share of attribution mass"
        )

    mask = np.asarray(mask).astype(bool)
    if mask.shape != saliency.shape:
        mask = _resample_mask(mask, saliency.shape)

    total = float(saliency.sum())
    if total <= 0:
        raise MaskError("saliency sums to zero; there is no attribution to apportion")

    coverage = float(mask.mean())
    if coverage <= 0:
        raise MaskError("mask is empty; segmentation failed for this image")
    if coverage >= 0.99:
        raise MaskError(
            "mask covers the entire frame; the fraction would be trivially 1.0 "
            "and carries no information"
        )

    fraction = float(saliency[mask].sum() / total)
    return AttributionScore(
        in_lung_fraction=fraction,
        lung_coverage=coverage,
        lift=fraction / coverage,
    )


def _resample_mask(mask: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    image = Image.fromarray(mask.astype(np.uint8) * 255, mode="L")
    resized = image.resize((shape[1], shape[0]), Image.NEAREST)
    return np.asarray(resized) > 127


def masked_out(image: np.ndarray, mask: np.ndarray, fill: float = 0.0) -> np.ndarray:
    """Blank everything outside the lungs, for the lung-masking ablation.

    Re-evaluate the model on these and accuracy should collapse toward chance.
    If it does not, the model was never using the lungs, and the headline
    number was measuring something else.
    """
    if mask.shape != image.shape[-2:]:
        mask = _resample_mask(np.asarray(mask).astype(bool), image.shape[-2:])
    return np.where(mask, image, fill).astype(image.dtype)


def coverage_summary(masks: list[np.ndarray]) -> dict[str, float]:
    """Distribution of lung coverage across a corpus.

    Worth checking before trusting any in-lung number: if coverage varies
    widely by source, then so does the null baseline, and comparing raw
    fractions across sources compares different things.
    """
    if not masks:
        return {}
    coverages = np.array([float(np.asarray(mask).astype(bool).mean()) for mask in masks])
    return {
        "mean": float(coverages.mean()),
        "std": float(coverages.std()),
        "min": float(coverages.min()),
        "max": float(coverages.max()),
    }
