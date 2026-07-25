"""The preprocessing contract, as data rather than code.

Train/serve preprocessing divergence is the most expensive bug this project
can ship: it produces a model that scores well offline and quietly degrades in
the app, with nothing in either codebase looking wrong. The usual cause is two
implementations of "resize and normalise" that disagree about one detail.

So the pipeline is declared as a serialisable spec that travels inside the
model artifact. The research package executes it with MONAI; the serving side
executes it with numpy and never imports this package. An equivalence test
asserts the two agree, which turns a silent drift into a failing build.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from enum import StrEnum
from pathlib import Path
from typing import Any

SPEC_VERSION = 1


class Windowing(StrEnum):
    """How raw intensities are mapped to a display range.

    DICOM carries a window centre and width chosen by the radiographer. Using
    it reproduces what a clinician saw; ignoring it and stretching the full
    range instead makes a subtle opacity disappear into the noise floor.
    """

    VOI_LUT = "voi_lut"
    PERCENTILE = "percentile"


class ResizeMode(StrEnum):
    PAD_TO_SQUARE = "pad_to_square"
    STRETCH = "stretch"


@dataclass(frozen=True)
class PreprocessingSpec:
    """Everything needed to turn a stored image into a model input."""

    target_size: int = 320
    windowing: Windowing = Windowing.VOI_LUT
    # Fallback percentiles when no VOI LUT is present, and the clip applied
    # after windowing. 1-99 keeps a hot pixel or a lead marker from
    # compressing the entire lung field into a few grey levels.
    clip_percentiles: tuple[float, float] = (1.0, 99.0)
    # Aspect ratio is preserved by padding. Stretching a chest to square
    # changes the cardiothoracic ratio, which is a measurement radiologists
    # actually take, and the distortion differs by source because source
    # aspect ratios differ - so squashing manufactures a confound.
    resize_mode: ResizeMode = ResizeMode.PAD_TO_SQUARE
    pad_value: float = 0.0
    # ImageNet backbones expect three channels; a radiograph has one.
    channels: int = 3
    normalise_mean: tuple[float, ...] = (0.485, 0.456, 0.406)
    normalise_std: tuple[float, ...] = (0.229, 0.224, 0.225)
    version: int = SPEC_VERSION
    notes: str = ""
    extra: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.target_size < 8:
            raise ValueError("target_size must be at least 8")
        low, high = self.clip_percentiles
        if not 0.0 <= low < high <= 100.0:
            raise ValueError(
                f"clip_percentiles must satisfy 0 <= low < high <= 100, got {(low, high)}"
            )
        if self.channels not in (1, 3):
            raise ValueError("channels must be 1 or 3")
        if len(self.normalise_mean) != self.channels:
            raise ValueError(
                f"normalise_mean has {len(self.normalise_mean)} values "
                f"but channels is {self.channels}"
            )
        if len(self.normalise_std) != self.channels:
            raise ValueError(
                f"normalise_std has {len(self.normalise_std)} values "
                f"but channels is {self.channels}"
            )
        if any(value <= 0 for value in self.normalise_std):
            raise ValueError("normalise_std values must be positive")

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["windowing"] = str(self.windowing)
        payload["resize_mode"] = str(self.resize_mode)
        payload["clip_percentiles"] = list(self.clip_percentiles)
        payload["normalise_mean"] = list(self.normalise_mean)
        payload["normalise_std"] = list(self.normalise_std)
        return payload

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> PreprocessingSpec:
        payload = dict(payload)
        stored = payload.pop("version", SPEC_VERSION)
        if stored > SPEC_VERSION:
            raise ValueError(
                f"spec version {stored} was written by a newer release than this one "
                f"(understands up to {SPEC_VERSION}); refusing to guess at its meaning"
            )
        return cls(
            windowing=Windowing(payload.pop("windowing", Windowing.VOI_LUT)),
            resize_mode=ResizeMode(payload.pop("resize_mode", ResizeMode.PAD_TO_SQUARE)),
            clip_percentiles=tuple(payload.pop("clip_percentiles", (1.0, 99.0))),
            normalise_mean=tuple(payload.pop("normalise_mean", (0.485, 0.456, 0.406))),
            normalise_std=tuple(payload.pop("normalise_std", (0.229, 0.224, 0.225))),
            version=stored,
            **payload,
        )

    def write(self, path: Path) -> None:
        """Persist beside the model artifact. This file is the contract."""
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")

    @classmethod
    def read(cls, path: Path) -> PreprocessingSpec:
        return cls.from_dict(json.loads(path.read_text(encoding="utf-8")))
