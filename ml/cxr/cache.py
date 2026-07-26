"""Precomputed deterministic preprocessing, as one memory-mapped array.

Half this corpus is DICOM at 1024x1024. Decoding it through the VOI LUT on
every epoch would make data loading, not the GPU, the thing that decides how
long training takes -- on four cores it is roughly an order of magnitude more
work than the forward and backward passes it feeds.

So the deterministic half of preprocessing runs once, and training reads the
result. Only the deterministic half: augmentation stays per-epoch, because
caching it would mean every epoch sees the same "random" crops, which is the
same as not augmenting.

The images are stored as uint8, which is a train/serve difference and therefore
needs justifying rather than assuming. Measured on 80 real radiographs from
both sources, the round trip costs 0.00875 in normalised units -- the exact
round-to-nearest bound, 1/(2*255*sigma). For scale, the augmentation this model
is explicitly trained to tolerate adds noise ten times larger, and the
MONAI-versus-Pillow resize divergence that killed the two-executor design was
229 times larger and bigger than the normalisation scale itself. Storing
float32 instead would cost 34 GB for this corpus against 2.1 GB, to remove an
error the model cannot notice. test_cache.py holds that number to account.
"""

from __future__ import annotations

import json
from collections.abc import Iterable
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from cxr.preprocessing import PreprocessingSpec, ResizeMode
from cxr.preprocessing import reference as ref

CACHE_VERSION = 1
IMAGES_FILE = "images.u8"
INDEX_FILE = "index.json"


class CacheError(RuntimeError):
    """Raised when a cache cannot be built or does not match what is asked of it."""


@dataclass(frozen=True)
class ImageCache:
    """A read-only view of precomputed images, addressable by image_id."""

    directory: Path
    spec: PreprocessingSpec
    image_ids: list[str]
    images: np.memmap

    def __post_init__(self) -> None:
        object.__setattr__(self, "_position", {name: i for i, name in enumerate(self.image_ids)})

    def __len__(self) -> int:
        return len(self.image_ids)

    def __contains__(self, image_id: str) -> bool:
        return image_id in self._position

    def windowed(self, image_id: str) -> np.ndarray:
        """The windowed, padded, resized image in [0, 1]. Not yet normalised."""
        try:
            position = self._position[image_id]
        except KeyError as error:
            raise CacheError(f"{image_id} is not in the cache at {self.directory}") from error
        return self.images[position].astype(np.float32) / 255.0

    def tensor(self, image_id: str) -> np.ndarray:
        """Channels-first and normalised: what the model actually consumes."""
        windowed = self.windowed(image_id)
        return ref.normalise(ref.to_channels(windowed, self.spec.channels), self.spec)


def deterministic_uint8(path: Path | str, spec: PreprocessingSpec) -> np.ndarray:
    """Everything in `reference.apply` up to the point normalisation begins."""
    image = ref.load_grayscale(Path(path))
    windowed = ref.window(image, spec)
    if spec.resize_mode is ResizeMode.PAD_TO_SQUARE:
        windowed = ref.pad_to_square(windowed, spec.pad_value)
    resized = ref.resize(windowed, spec.target_size)
    return np.round(np.clip(resized, 0.0, 1.0) * 255.0).astype(np.uint8)


def _one(job: tuple[int, str, str]) -> tuple[int, np.ndarray]:
    position, path, spec_json = job
    spec = PreprocessingSpec.from_dict(json.loads(spec_json))
    return position, deterministic_uint8(Path(path), spec)


def build(
    frame: pd.DataFrame,
    directory: Path | str,
    *,
    spec: PreprocessingSpec,
    image_roots: dict[str, Path],
    workers: int = 4,
) -> ImageCache:
    """Preprocess every row once and write the result to `directory`."""
    directory = Path(directory)
    missing = sorted(set(frame["source"]) - set(image_roots))
    if missing:
        raise CacheError(f"no image root given for source(s): {missing}")

    directory.mkdir(parents=True, exist_ok=True)
    image_ids = frame["image_id"].tolist()
    side = spec.target_size
    images = np.memmap(
        directory / IMAGES_FILE, dtype=np.uint8, mode="w+", shape=(len(frame), side, side)
    )

    spec_json = json.dumps(spec.to_dict())
    jobs = [
        (position, str(Path(image_roots[row.source]) / row.path), spec_json)
        for position, row in enumerate(frame.itertuples(index=False))
    ]

    # Decoding is CPU-bound in native code, so processes rather than threads.
    with ProcessPoolExecutor(max_workers=workers) as pool:
        for position, pixels in pool.map(_one, jobs, chunksize=16):
            images[position] = pixels

    images.flush()
    (directory / INDEX_FILE).write_text(
        json.dumps(
            {"version": CACHE_VERSION, "spec": spec.to_dict(), "image_ids": image_ids}, indent=2
        ),
        encoding="utf-8",
    )
    return load(directory)


def load(directory: Path | str, *, expect: PreprocessingSpec | None = None) -> ImageCache:
    """Open an existing cache, refusing one built under a different spec.

    Reusing a cache whose spec has drifted is the quiet version of training on
    one preprocessing and serving another, so it is an error rather than a
    warning.
    """
    directory = Path(directory)
    index_path = directory / INDEX_FILE
    if not index_path.exists():
        raise CacheError(f"no cache index at {index_path}; run `cxr cache` first")

    index = json.loads(index_path.read_text(encoding="utf-8"))
    if index.get("version") != CACHE_VERSION:
        raise CacheError(
            f"cache at {directory} is version {index.get('version')}, this build expects "
            f"{CACHE_VERSION}; rebuild it"
        )

    spec = PreprocessingSpec.from_dict(index["spec"])
    if expect is not None and spec != expect:
        raise CacheError(
            f"cache at {directory} was built under a different preprocessing spec; rebuild it "
            "rather than training on one preprocessing and serving another"
        )

    image_ids = list(index["image_ids"])
    side = spec.target_size
    images = np.memmap(
        directory / IMAGES_FILE, dtype=np.uint8, mode="r", shape=(len(image_ids), side, side)
    )
    return ImageCache(directory=directory, spec=spec, image_ids=image_ids, images=images)


def missing_from(cache: ImageCache, image_ids: Iterable[str]) -> list[str]:
    """Which of these rows the cache cannot serve. Empty is the only safe answer."""
    return [image_id for image_id in image_ids if image_id not in cache]
