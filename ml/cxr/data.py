"""Torch datasets over a split manifest.

The split column is the only thing that decides what a model sees. Nothing here
reshuffles, resamples across splits, or falls back to "just use everything" when
a split is empty -- the gates spent their whole existence establishing that
those splits mean something.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from cxr.cache import ImageCache, missing_from
from cxr.manifest import Label
from cxr.preprocessing import PreprocessingSpec
from cxr.preprocessing import reference as ref

# Fixed, not derived from whatever happens to be in the manifest. A model whose
# output index 0 means "covid" in one run and "normal" in the next is a model
# whose saved weights cannot be trusted against a saved threshold.
CLASSES: tuple[str, ...] = (str(Label.NORMAL), str(Label.PNEUMONIA), str(Label.COVID))
CLASS_INDEX = {name: index for index, name in enumerate(CLASSES)}


class SplitError(ValueError):
    """Raised when a requested split cannot be served."""


def class_weights(frame: pd.DataFrame) -> np.ndarray:
    """Inverse-frequency weights in CLASSES order.

    Deliberately not resampling. Oversampling the minority class duplicates
    images inside the training split, which is the same repetition G2 exists to
    remove between splits, and it inflates the effective epoch length in a way
    that makes runs incomparable.
    """
    counts = frame["label"].value_counts()
    frequencies = np.array([counts.get(name, 0) for name in CLASSES], dtype=np.float64)
    if (frequencies == 0).any():
        absent = [name for name, count in zip(CLASSES, frequencies, strict=True) if count == 0]
        raise SplitError(f"class(es) {absent} have no rows; a weight for them is undefined")
    weights = frequencies.sum() / (len(CLASSES) * frequencies)
    return weights.astype(np.float32)


class RadiographDataset:
    """One split, served either from a cache or straight from disk.

    `augment` is the only difference between the training and evaluation views,
    and it is a constructor argument rather than a mode flag so that an
    evaluation loader cannot be handed an augmenting pipeline by accident.
    """

    def __init__(
        self,
        frame: pd.DataFrame,
        *,
        spec: PreprocessingSpec,
        cache: ImageCache | None = None,
        image_roots: dict[str, Path] | None = None,
        augment=None,
    ) -> None:
        if cache is None and not image_roots:
            raise SplitError("give either a cache or image roots; there is nothing to read from")
        if cache is not None:
            absent = missing_from(cache, frame["image_id"])
            if absent:
                raise SplitError(
                    f"{len(absent)} rows are not in the cache (first: {absent[0]}); "
                    "rebuild it against this manifest rather than silently skipping them"
                )

        self.frame = frame.reset_index(drop=True)
        self.spec = spec
        self.cache = cache
        self.image_roots = image_roots or {}
        self.augment = augment
        self.labels = np.array(
            [CLASS_INDEX[label] for label in self.frame["label"]], dtype=np.int64
        )

    def __len__(self) -> int:
        return len(self.frame)

    def _windowed(self, position: int) -> np.ndarray:
        row = self.frame.iloc[position]
        if self.cache is not None:
            return self.cache.windowed(row["image_id"])
        root = self.image_roots.get(row["source"])
        if root is None:
            raise SplitError(f"no image root for source {row['source']!r}")
        from cxr.cache import deterministic_uint8

        return deterministic_uint8(Path(root) / row["path"], self.spec).astype(np.float32) / 255.0

    def __getitem__(self, position: int):
        image = ref.normalise(
            ref.to_channels(self._windowed(position), self.spec.channels), self.spec
        )
        if self.augment is not None:
            image = np.asarray(self.augment(image), dtype=np.float32)
        return image, int(self.labels[position])


def for_split(
    frame: pd.DataFrame,
    split: str,
    *,
    spec: PreprocessingSpec,
    cache: ImageCache | None = None,
    image_roots: dict[str, Path] | None = None,
    augment=None,
) -> RadiographDataset:
    """The rows of one split, refusing to invent data when there are none."""
    if "split" not in frame.columns:
        raise SplitError("manifest has no split column; run `cxr split` first")
    rows = frame[frame["split"] == split]
    if rows.empty:
        raise SplitError(f"split {split!r} is empty; splits present: {sorted(set(frame['split']))}")
    return RadiographDataset(
        rows, spec=spec, cache=cache, image_roots=image_roots, augment=augment
    )
