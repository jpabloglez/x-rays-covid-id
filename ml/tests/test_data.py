"""Datasets over a split manifest."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from cxr import cache as image_cache
from cxr.data import CLASSES, RadiographDataset, SplitError, class_weights, for_split
from cxr.manifest import Label, LabelProvenance, View
from cxr.preprocessing import PreprocessingSpec
from PIL import Image


def _row(image_id, path, label, split):
    return {
        "image_id": image_id,
        "source": "covid_radiography",
        "path": path,
        "label": str(label),
        "label_raw": str(label),
        "label_provenance": str(LabelProvenance.RADIOLOGIST),
        "patient_id": f"covid_radiography:{image_id}",
        "study_id": None,
        "view": str(View.PA),
        "age": 50.0,
        "sex": "F",
        "sha256": f"sha-{image_id}",
        "phash": "0" * 64,
        "width": 96,
        "height": 96,
        "mask_path": None,
        "split": split,
    }


LABELS = (Label.NORMAL, Label.PNEUMONIA, Label.COVID)


@pytest.fixture
def corpus(tmp_path):
    root = tmp_path / "images"
    (root / "images").mkdir(parents=True)
    rng = np.random.default_rng(0)
    rows = []
    for index in range(12):
        name = f"chest{index}.png"
        Image.fromarray((rng.random((96, 80)) * 220).astype(np.uint8), mode="L").save(
            root / "images" / name
        )
        rows.append(
            _row(f"c{index}", f"images/{name}", LABELS[index % 3], "train" if index < 9 else "val")
        )
    return root, pd.DataFrame(rows)


SPEC = PreprocessingSpec(target_size=32)


def test_class_order_is_fixed_not_derived():
    """Output index 0 must mean the same class in every run, or a saved
    threshold applied to saved weights is meaningless."""
    assert CLASSES == ("normal", "pneumonia", "covid")


def test_dataset_serves_images_and_integer_labels(corpus):
    root, frame = corpus
    dataset = for_split(frame, "train", spec=SPEC, image_roots={"covid_radiography": root})
    image, label = dataset[0]
    assert image.shape == (3, 32, 32)
    assert image.dtype == np.float32
    assert label in range(3)


def test_split_selection_never_falls_back_to_everything(corpus):
    _, frame = corpus
    with pytest.raises(SplitError, match="empty"):
        for_split(frame, "test", spec=SPEC, image_roots={"covid_radiography": "."})


def test_a_manifest_without_splits_is_rejected(corpus):
    root, frame = corpus
    with pytest.raises(SplitError, match="no split column"):
        for_split(
            frame.drop(columns=["split"]), "train", spec=SPEC,
            image_roots={"covid_radiography": root},
        )


def test_a_cache_missing_rows_is_an_error_not_a_silent_skip(corpus, tmp_path):
    """Dropping uncached rows would shrink a split without anyone being told."""
    root, frame = corpus
    built = image_cache.build(
        frame.head(4), tmp_path / "cache", spec=SPEC,
        image_roots={"covid_radiography": root}, workers=1,
    )
    with pytest.raises(SplitError, match="not in the cache"):
        RadiographDataset(frame, spec=SPEC, cache=built)


def test_cached_and_uncached_reads_agree(corpus, tmp_path):
    root, frame = corpus
    built = image_cache.build(
        frame, tmp_path / "cache", spec=SPEC,
        image_roots={"covid_radiography": root}, workers=1,
    )
    from_cache = RadiographDataset(frame, spec=SPEC, cache=built)
    from_disk = RadiographDataset(frame, spec=SPEC, image_roots={"covid_radiography": root})
    assert np.array_equal(from_cache[3][0], from_disk[3][0])


def test_class_weights_are_inverse_frequency_in_class_order():
    frame = pd.DataFrame(
        {"label": ["normal"] * 60 + ["pneumonia"] * 30 + ["covid"] * 10}
    )
    weights = class_weights(frame)
    assert weights.argmax() == CLASSES.index("covid")
    assert weights[CLASSES.index("normal")] < weights[CLASSES.index("pneumonia")]


def test_a_missing_class_has_no_defined_weight():
    frame = pd.DataFrame({"label": ["normal"] * 10 + ["pneumonia"] * 10})
    with pytest.raises(SplitError, match="covid"):
        class_weights(frame)


def test_a_dataset_with_nowhere_to_read_from_is_rejected(corpus):
    _, frame = corpus
    with pytest.raises(SplitError, match="nothing to read from"):
        RadiographDataset(frame, spec=SPEC)
