"""The reference executor: numpy and PIL only.

This is the definition of what the spec means. It deliberately depends on
nothing heavier than numpy and Pillow, so the serving side can reimplement it
line for line without importing this package or pulling in torch.

When the MONAI executor and this one disagree, this one is right and MONAI is
misconfigured — because this is what the deployed model will actually be fed.
"""

from __future__ import annotations

import numpy as np
from PIL import Image

from cxr.preprocessing.spec import PreprocessingSpec, ResizeMode

# Pillow modes whose samples do not fit in a byte. `convert("L")` clips these
# at 255 rather than rescaling, which turns a 12-bit radiograph into a white
# rectangle. Part of the spec's definition, so a reimplementation has to
# reproduce it: get this wrong and the model is served a different image than
# it was trained on.
DEEP_MODES = frozenset({"I", "I;16", "I;16B", "I;16L", "I;16N", "F"})


def window(image: np.ndarray, spec: PreprocessingSpec) -> np.ndarray:
    """Clip to the configured percentiles and scale to [0, 1].

    Percentiles rather than min/max: a single saturated pixel, a lead marker
    or a burnt-in annotation would otherwise compress the whole lung field
    into a handful of grey levels.
    """
    low_percentile, high_percentile = spec.clip_percentiles
    low = float(np.percentile(image, low_percentile))
    high = float(np.percentile(image, high_percentile))
    if high <= low:
        # A uniform image. Return mid-grey rather than dividing by zero; the
        # OOD gate in Phase D is what should reject this, not preprocessing.
        return np.full(image.shape, 0.5, dtype=np.float32)
    clipped = np.clip(image, low, high)
    return ((clipped - low) / (high - low)).astype(np.float32)


def pad_to_square(image: np.ndarray, pad_value: float) -> np.ndarray:
    """Centre the image in a square canvas, preserving aspect ratio."""
    height, width = image.shape
    side = max(height, width)
    if height == width:
        return image
    canvas = np.full((side, side), pad_value, dtype=image.dtype)
    top = (side - height) // 2
    left = (side - width) // 2
    canvas[top : top + height, left : left + width] = image
    return canvas


def resize(image: np.ndarray, target_size: int) -> np.ndarray:
    """Bilinear resize through Pillow.

    Pillow rather than a hand-rolled kernel because the serving side will have
    Pillow anyway, and two implementations of bilinear resampling that differ
    at the edges is exactly the kind of drift this module exists to prevent.
    """
    if image.shape == (target_size, target_size):
        return image
    pil = Image.fromarray(image, mode="F")
    resized = pil.resize((target_size, target_size), Image.BILINEAR)
    return np.asarray(resized, dtype=np.float32)


def to_channels(image: np.ndarray, channels: int) -> np.ndarray:
    """Replicate grayscale across channels, returning CHW."""
    if channels == 1:
        return image[np.newaxis, ...]
    return np.repeat(image[np.newaxis, ...], channels, axis=0)


def normalise(image: np.ndarray, spec: PreprocessingSpec) -> np.ndarray:
    mean = np.asarray(spec.normalise_mean, dtype=np.float32).reshape(-1, 1, 1)
    std = np.asarray(spec.normalise_std, dtype=np.float32).reshape(-1, 1, 1)
    return ((image - mean) / std).astype(np.float32)


def apply(image: np.ndarray, spec: PreprocessingSpec) -> np.ndarray:
    """Run the full spec over a single-channel float array, returning CHW.

    Order is load-bearing. Windowing happens on the native resolution, because
    percentiles computed after resampling are percentiles of interpolated
    pixels. Padding happens before resizing, because doing it after would
    resize the real content to the target and then grow the canvas past it.
    """
    if image.ndim != 2:
        raise ValueError(f"expected a single-channel 2-D image, got shape {image.shape}")

    windowed = window(image.astype(np.float32), spec)
    if spec.resize_mode is ResizeMode.PAD_TO_SQUARE:
        windowed = pad_to_square(windowed, spec.pad_value)
    resized = resize(windowed, spec.target_size)
    stacked = to_channels(resized, spec.channels)
    return normalise(stacked, spec)


def load_grayscale(path) -> np.ndarray:
    """Read a stored image as a 2-D float array.

    DICOM goes through the dedicated decoder so the VOI LUT and
    PhotometricInterpretation are honoured; everything else through Pillow.
    """
    from pathlib import Path

    path = Path(path)
    if path.suffix.lower() in {".dcm", ".dicom"}:
        from cxr.preprocessing.dicom import decode

        return decode(path)
    with Image.open(path) as image:
        if image.mode in DEEP_MODES:
            # Not convert("L"): it clips at 255, and BIMCV's 12-bit pixel data
            # runs to about 4095 with its darkest pixel already in the
            # hundreds, so every radiograph would arrive as uniform white. The
            # raw values go through untouched because `window` rescales by
            # percentile immediately afterwards, which is both the correct
            # place for it and more robust than anything done here.
            return np.asarray(image, dtype=np.float32)
        return np.asarray(image.convert("L"), dtype=np.float32)
