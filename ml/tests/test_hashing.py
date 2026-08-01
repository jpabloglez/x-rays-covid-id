"""Hashing, and specifically that the banded clustering is exact."""

from __future__ import annotations

import numpy as np
import pytest
from cxr.hashing import PHASH_BITS, cluster, dhash, hamming, sha256_file
from PIL import Image


def test_sha256_matches_between_identical_files(tmp_path):
    first, second = tmp_path / "a.bin", tmp_path / "b.bin"
    first.write_bytes(b"radiograph")
    second.write_bytes(b"radiograph")
    assert sha256_file(first) == sha256_file(second)


def test_dhash_survives_recompression(tmp_path):
    """The point of a perceptual hash: JPEG round-tripping must not change it.

    This is exactly what defeats SHA-256 on the aggregate COVID collections.
    """
    rng = np.random.default_rng(0)
    pixels = (rng.random((64, 64)) * 255).astype(np.uint8)
    original = Image.fromarray(pixels, mode="L")

    path = tmp_path / "recompressed.jpg"
    original.save(path, quality=80)
    with Image.open(path) as recompressed:
        assert hamming(dhash(original), dhash(recompressed)) <= 4


def test_dhash_survives_resizing():
    rng = np.random.default_rng(1)
    pixels = (rng.random((128, 128)) * 255).astype(np.uint8)
    original = Image.fromarray(pixels, mode="L")
    resized = original.resize((96, 96), Image.BICUBIC)
    assert hamming(dhash(original), dhash(resized)) <= 8


def _twelve_bit(seed, side=64):
    """A 12-bit radiograph in a 16-bit container, as BIMCV ships them.

    Smooth, and entirely above 255. Both properties matter. Random noise at
    this depth survives clipping -- the few percent of pixels that land under
    255 keep enough gradient to hash -- so a noise fixture passes whether the
    bug is present or not. A real radiograph is smooth and its darkest pixel
    still sits in the hundreds, which is what makes clipping total.
    """
    rng = np.random.default_rng(seed)
    coarse = Image.fromarray((rng.random((8, 8)) * 255).astype(np.uint8), mode="L")
    smooth = np.asarray(coarse.resize((side, side), Image.BICUBIC), dtype=np.float64) / 255.0
    return Image.fromarray((1000 + smooth * 3095).astype(np.uint16))


def test_a_sixteen_bit_image_does_not_hash_to_nothing():
    """`convert("L")` clips at 255, so 12-bit pixel data arrives as pure white.

    A uniform image has no horizontal gradients, so its hash is 256 zero bits
    -- and every such image then sits within zero bits of every other. On
    BIMCV that collapsed 501 radiographs from 320 different patients into one
    "duplicate" group, which the splitter would have had to keep together.
    """
    bits = bin(int(dhash(_twelve_bit(0)), 16)).count("1")
    assert bits > PHASH_BITS // 4, f"only {bits} of {PHASH_BITS} bits set"


def test_sixteen_bit_images_of_different_patients_stay_apart():
    assert hamming(dhash(_twelve_bit(0)), dhash(_twelve_bit(1))) > 12


def test_rescaling_the_deep_modes_leaves_eight_bit_hashes_untouched():
    """The 8-bit collections were hashed before this path existed, and their
    duplicate groups are a published result; changing them silently would
    invalidate it."""
    rng = np.random.default_rng(3)
    pixels = (rng.random((64, 64)) * 255).astype(np.uint8)
    image = Image.fromarray(pixels, mode="L")
    assert dhash(image) == dhash(image.convert("L"))


def test_a_uniform_deep_image_hashes_without_dividing_by_zero():
    blank = Image.fromarray(np.full((32, 32), 2000, dtype=np.uint16))
    assert int(dhash(blank), 16) == 0


def test_unrelated_images_are_far_apart():
    rng = np.random.default_rng(2)
    left = Image.fromarray((rng.random((64, 64)) * 255).astype(np.uint8), mode="L")
    right = Image.fromarray((rng.random((64, 64)) * 255).astype(np.uint8), mode="L")
    assert hamming(dhash(left), dhash(right)) > 12


def test_identical_hashes_cluster_together():
    items = [("a", "ffffffffffffffff"), ("b", "ffffffffffffffff"), ("c", "0000000000000000")]
    assignments = cluster(items, threshold=4)
    assert assignments["a"] == assignments["b"]
    assert assignments["a"] != assignments["c"]


def test_clustering_is_transitive():
    """Chained near-duplicates land in one cluster, which is what we want.

    An image resized then recompressed may be 3 bits from the original and 3
    from the intermediate; all three must be kept on the same side of a split.
    """
    items = [
        ("a", f"{0b0000:016x}"),
        ("b", f"{0b0011:016x}"),
        ("c", f"{0b1111:016x}"),
    ]
    assignments = cluster(items, threshold=2)
    assert assignments["a"] == assignments["b"] == assignments["c"]


@pytest.mark.parametrize("threshold", [0, 1, 3, 6, 10])
def test_banding_never_misses_a_pair_inside_the_threshold(threshold):
    """The banded bucketing must be exact, not approximate.

    A missed pair is a leaked duplicate, so this is checked against brute force
    on random hashes rather than assumed from the pigeonhole argument alone.
    """
    rng = np.random.default_rng(threshold)
    values = [int(rng.integers(0, 2**63)) for _ in range(120)]
    items = [(str(index), f"{value:016x}") for index, value in enumerate(values)]

    assignments = cluster(items, threshold=threshold)

    for i in range(len(values)):
        for j in range(i + 1, len(values)):
            distance = (values[i] ^ values[j]).bit_count()
            if distance <= threshold:
                assert assignments[str(i)] == assignments[str(j)], (
                    f"pair ({i}, {j}) at distance {distance} was not clustered"
                )


def test_empty_input_clusters_to_nothing():
    assert cluster([], threshold=4) == {}


def test_negative_threshold_is_rejected():
    with pytest.raises(ValueError, match="non-negative"):
        cluster([("a", "0" * 16)], threshold=-1)


def test_hash_is_the_declared_width():
    image = Image.fromarray(np.zeros((16, 16), dtype=np.uint8), mode="L")
    assert len(dhash(image)) == PHASH_BITS // 4
