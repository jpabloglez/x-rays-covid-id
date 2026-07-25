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

PHASH_BITS = 64
_DHASH_SIDE = 8

# Chest radiographs share their gross anatomy, so their hashes sit closer
# together than natural images do and a threshold tuned on photographs will
# merge unrelated patients into one cluster. Calibrate on the real corpus
# before trusting it: cluster at several thresholds, and if a cluster ever
# contains two different patient_ids that are not a known duplicate pair, the
# threshold is too loose. Erring loose is the safer direction — an
# over-merged cluster costs training rows, an under-merged one leaks.
DEFAULT_THRESHOLD_BITS = 6


def sha256_file(path: Path, chunk_size: int = 1 << 20) -> str:
    """SHA-256 of the raw bytes, streamed so a large DICOM does not sit in RAM."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def dhash(image: Image.Image) -> str:
    """Difference hash: 64 bits of horizontal-gradient sign, as 16 hex chars.

    Chosen over average hash because it survives the global brightness and
    contrast changes that re-encoding introduces, and over pHash because it
    needs no DCT and stays dependency-light enough to run in CI.
    """
    grayscale = image.convert("L").resize((_DHASH_SIDE + 1, _DHASH_SIDE), Image.LANCZOS)
    pixels = np.asarray(grayscale, dtype=np.int16)
    bits = pixels[:, 1:] > pixels[:, :-1]
    value = 0
    for bit in bits.flatten():
        value = (value << 1) | int(bit)
    return f"{value:016x}"


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
