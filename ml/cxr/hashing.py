"""Exact and near-duplicate detection.

Exact duplicates are cheap: SHA-256 of the file bytes. Near-duplicates are the
problem worth solving, because the public COVID collections are aggregates —
the same radiograph appears in several of them, re-encoded, re-cropped or
resized, and lands on both sides of a split under different filenames.
"""

from __future__ import annotations

import hashlib
from collections.abc import Iterable, Iterator
from pathlib import Path

import numpy as np
from PIL import Image

PHASH_BITS = 256
_DHASH_SIDE = 16

# Modes whose samples do not fit in a byte. PIL converts these to "L" by
# clipping at 255 rather than rescaling, so a 12-bit radiograph stored in
# `I;16` arrives as uniform white.
_DEEP_MODES = frozenset({"I", "I;16", "I;16B", "I;16L", "I;16N", "F"})


def _grayscale(image: Image.Image) -> Image.Image:
    """8-bit grayscale, rescaled rather than clipped.

    BIMCV ships 12-bit pixel data in 16-bit PNGs, values running to about 4095.
    `convert("L")` clips every one of them to 255, and a uniform image has no
    horizontal gradients at all, so its hash is 256 zero bits. Every such image
    then sits within zero bits of every other, and 501 radiographs from 320
    different patients collapsed into a single "duplicate" group.

    Rescaling by the observed range is safe for this hash specifically: dHash
    compares neighbouring pixels, and a monotonic rescale cannot change which
    of two neighbours is brighter. An 8-bit image takes the original path
    untouched, so hashes already computed for the 8-bit collections stand.
    """
    if image.mode not in _DEEP_MODES:
        return image.convert("L")

    pixels = np.asarray(image, dtype=np.float64)
    low, high = float(pixels.min()), float(pixels.max())
    if high <= low:
        # Genuinely blank. Hashing it is pointless but must not divide by zero.
        return Image.fromarray(np.zeros(pixels.shape, dtype=np.uint8), mode="L")
    scaled = (pixels - low) * (255.0 / (high - low))
    return Image.fromarray(scaled.astype(np.uint8), mode="L")

# Calibrated on the real 30k corpus rather than guessed.
#
# At 64 bits this hash was unusable here. Chest radiographs share their gross
# anatomy, and a 64-bit signature could not separate them: 9,026 of 30,016
# images collided exactly while only 54 were byte-identical, and the nearest
# *distinct* pair sat 1 bit apart, so any threshold at all chained the corpus
# into a single cluster covering 94% of it.
#
# At 256 bits the distribution is cleanly bimodal: true duplicates at distance
# 0, and the nearest distinct pair 23 bits away. A threshold in the middle of
# that gap separates them with margin to spare.
DEFAULT_THRESHOLD_BITS = 10


def sha256_file(path: Path, chunk_size: int = 1 << 20) -> str:
    """SHA-256 of the raw bytes, streamed so a large DICOM does not sit in RAM."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def dhash(image: Image.Image) -> str:
    """Difference hash: 256 bits of horizontal-gradient sign, as 64 hex chars.

    Chosen over average hash because it survives the global brightness and
    contrast changes that re-encoding introduces, and over pHash because it
    needs no DCT and stays dependency-light enough to run in CI.
    """
    grayscale = _grayscale(image).resize((_DHASH_SIDE + 1, _DHASH_SIDE), Image.LANCZOS)
    pixels = np.asarray(grayscale, dtype=np.int16)
    bits = pixels[:, 1:] > pixels[:, :-1]
    value = 0
    for bit in bits.flatten():
        value = (value << 1) | int(bit)
    return f"{value:0{PHASH_BITS // 4}x}"


def dhash_file(path: Path) -> str:
    with Image.open(path) as image:
        return dhash(image)


def hamming(left: str, right: str) -> int:
    """Bit distance between two hex-encoded hashes."""
    return (int(left, 16) ^ int(right, 16)).bit_count()


def _bands(value: int, band_count: int) -> Iterator[tuple[int, int]]:
    """Split a hash into band_count near-equal slices, yielding (index, value).

    With band_count = threshold + 1, two hashes within `threshold` bits must
    agree exactly on at least one band: there are more bands than differing
    bits, so by pigeonhole one band contains none of them. That makes the
    bucketing below exact — it never misses a pair inside the threshold.
    """
    edges = np.linspace(0, PHASH_BITS, band_count + 1).astype(int)
    for index in range(band_count):
        low, high = int(edges[index]), int(edges[index + 1])
        width = high - low
        mask = (1 << width) - 1
        yield index, (value >> (PHASH_BITS - high)) & mask


def cluster(
    items: Iterable[tuple[str, str]], threshold: int = 6
) -> dict[str, int]:
    """Group ids whose perceptual hashes are within `threshold` bits.

    Returns id -> cluster id. Pairwise comparison is O(n²) and unusable on a
    six-figure corpus, so candidates are bucketed by band first and only
    compared within a bucket; the banding is exact for the given threshold.
    """
    if threshold < 0:
        raise ValueError("threshold must be non-negative")
    entries = [(item_id, int(digest, 16)) for item_id, digest in items]
    if not entries:
        return {}

    band_count = threshold + 1
    buckets: dict[tuple[int, int], list[int]] = {}
    for position, (_, digest) in enumerate(entries):
        for band_index, band_value in _bands(digest, band_count):
            buckets.setdefault((band_index, band_value), []).append(position)

    parent = list(range(len(entries)))

    def find(node: int) -> int:
        while parent[node] != node:
            parent[node] = parent[parent[node]]
            node = parent[node]
        return node

    def union(left: int, right: int) -> None:
        left_root, right_root = find(left), find(right)
        if left_root != right_root:
            parent[max(left_root, right_root)] = min(left_root, right_root)

    for candidates in buckets.values():
        if len(candidates) < 2:
            continue
        for offset, first in enumerate(candidates):
            for second in candidates[offset + 1 :]:
                if find(first) == find(second):
                    continue
                if (entries[first][1] ^ entries[second][1]).bit_count() <= threshold:
                    union(first, second)

    return {entries[position][0]: find(position) for position in range(len(entries))}
