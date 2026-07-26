"""The cache, and the measurement that justifies storing it as uint8."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from cxr import cache as image_cache
from cxr.cache import CacheError
from cxr.manifest import Label, LabelProvenance, View
from cxr.preprocessing import PreprocessingSpec
from cxr.preprocessing import reference as ref
from PIL import Image


def _row(image_id, path, label=Label.NORMAL, source="covid_radiography"):
    return {
        "image_id": image_id,
        "source": source,
        "path": path,
        "label": str(label),
        "label_raw": str(label),
        "label_provenance": str(LabelProvenance.RADIOLOGIST),
        "patient_id": f"{source}:{image_id}",
        "study_id": None,
        "view": str(View.PA),
        "age": 50.0,
        "sex": "F",
        "sha256": f"sha-{image_id}",
        "phash": f"{abs(hash(image_id)):064x}"[:64],
        "width": 96,
        "height": 96,
        "mask_path": None,
    }


@pytest.fixture
def corpus(tmp_path):
    root = tmp_path / "images"
    rows = []
    rng = np.random.default_rng(0)
    for index in range(6):
        name = f"chest{index}.png"
        (root / "images").mkdir(parents=True, exist_ok=True)
        pixels = (rng.random((96, 80)) * 200 + 20).astype(np.uint8)
        Image.fromarray(pixels, mode="L").save(root / "images" / name)
        rows.append(_row(f"c{index}", f"images/{name}"))
    return root, pd.DataFrame(rows)


def test_cache_round_trips_every_row(corpus, tmp_path):
    root, frame = corpus
    spec = PreprocessingSpec(target_size=64)
    built = image_cache.build(
        frame, tmp_path / "cache", spec=spec, image_roots={"covid_radiography": root}, workers=2
    )
    assert len(built) == len(frame)
    assert all(image_id in built for image_id in frame["image_id"])
    assert built.tensor("c0").shape == (3, 64, 64)


def test_uint8_storage_costs_less_than_the_augmentation_it_lives_under(corpus, tmp_path):
    """The number in cache.py's docstring, held to account.

    Storing the cache as uint8 is a train/serve difference, and the two-executor
    design was retired over exactly that. The difference is the magnitude: this
    one is the round-to-nearest bound, an order of magnitude below the noise
    augmentation deliberately adds, where MONAI-versus-Pillow was larger than
    the normalisation scale itself. If this ever exceeds the augmentation noise,
    the cache needs to hold float16 and this test should say so.
    """
    root, frame = corpus
    spec = PreprocessingSpec(target_size=64)
    built = image_cache.build(
        frame, tmp_path / "cache", spec=spec, image_roots={"covid_radiography": root}, workers=1
    )

    augmentation_noise = 0.02 / spec.normalise_std[0]
    worst = 0.0
    for row in frame.to_dict("records"):
        exact = ref.apply(ref.load_grayscale(root / row["path"]), spec)
        worst = max(worst, float(np.abs(built.tensor(row["image_id"]) - exact).max()))

    assert worst < augmentation_noise / 5
    # The bound is deterministic, so pin it rather than leaving it open-ended.
    assert worst == pytest.approx(1.0 / (2 * 255 * spec.normalise_std[0]), rel=0.05)


def test_loading_refuses_a_cache_built_under_a_different_spec(corpus, tmp_path):
    """Otherwise a stale cache silently trains on preprocessing that serving
    will never reproduce."""
    root, frame = corpus
    image_cache.build(
        frame,
        tmp_path / "cache",
        spec=PreprocessingSpec(target_size=64),
        image_roots={"covid_radiography": root},
        workers=1,
    )
    with pytest.raises(CacheError, match="different preprocessing spec"):
        image_cache.load(tmp_path / "cache", expect=PreprocessingSpec(target_size=128))


def test_building_without_a_root_for_every_source_is_an_error(corpus, tmp_path):
    root, frame = corpus
    frame = pd.concat([frame, pd.DataFrame([_row("r0", "x.dcm", source="rsna_pneumonia")])])
    with pytest.raises(CacheError, match="rsna_pneumonia"):
        image_cache.build(
            frame,
            tmp_path / "cache",
            spec=PreprocessingSpec(target_size=32),
            image_roots={"covid_radiography": root},
        )


def test_a_missing_cache_says_what_to_run(tmp_path):
    with pytest.raises(CacheError, match="cxr cache"):
        image_cache.load(tmp_path / "nothing")
