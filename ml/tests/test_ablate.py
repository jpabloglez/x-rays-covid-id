"""The ablation dataset: masks must land in register with the images."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from cxr.ablate import BASELINE, LUNGS_BLANKED, LUNGS_ONLY, AblationDataset
from cxr.data import SplitError
from cxr.manifest import Label, LabelProvenance, View
from cxr.preprocessing import PreprocessingSpec
from PIL import Image

SPEC = PreprocessingSpec(target_size=32)


def _row(image_id, path, mask_path, label=Label.COVID):
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
        "width": 40,
        "height": 80,
        "mask_path": mask_path,
        "split": "test",
    }


@pytest.fixture
def corpus(tmp_path):
    """Deliberately non-square (80x40), which is where registration breaks.

    The mask covers the top half of the original image. After padding to square
    and resizing, it must still cover the top half of the anatomy -- not the
    top half of the padded canvas.
    """
    root = tmp_path / "src"
    (root / "images").mkdir(parents=True)
    (root / "masks").mkdir(parents=True)

    image = np.zeros((80, 40), dtype=np.uint8)
    image[:40] = 200  # bright top half
    image[40:] = 60  # dark bottom half
    Image.fromarray(image, mode="L").save(root / "images" / "c0.png")

    mask = np.zeros((80, 40), dtype=np.uint8)
    mask[:40] = 255  # "lungs" = the bright top half
    Image.fromarray(mask, mode="L").save(root / "masks" / "c0.png")

    frame = pd.DataFrame([_row("c0", "images/c0.png", "masks/c0.png")])
    return root, frame


def _dataset(corpus, variant):
    root, frame = corpus
    return AblationDataset(
        frame,
        spec=SPEC,
        variant=variant,
        mask_roots={"covid_radiography": root},
        image_roots={"covid_radiography": root},
    )


def test_lungs_only_keeps_the_masked_region(corpus):
    image, _ = _dataset(corpus, LUNGS_ONLY)[0]
    top, bottom = image[0, :16, :], image[0, 16:, :]
    # The bright half survives; everything else is blanked to the fill value.
    assert top.max() > bottom.max()
    assert float(np.abs(bottom).max()) == pytest.approx(float(np.abs(bottom).min()), abs=1e-6)


def test_lungs_blanked_removes_exactly_that_region(corpus):
    image, _ = _dataset(corpus, LUNGS_BLANKED)[0]
    baseline, _ = _dataset(corpus, BASELINE)[0]
    changed = ~np.isclose(image[0], baseline[0])
    # Only the top half changed, and it actually did change.
    assert changed[:16, :].any()
    assert not changed[17:, :].any()


def test_the_two_ablations_are_complements(corpus):
    """Together they must cover the image: anything neither keeps is a
    registration bug, and would silently weaken both measurements."""
    only, _ = _dataset(corpus, LUNGS_ONLY)[0]
    blanked, _ = _dataset(corpus, LUNGS_BLANKED)[0]
    baseline, _ = _dataset(corpus, BASELINE)[0]

    kept_by_one = np.isclose(only[0], baseline[0]) | np.isclose(blanked[0], baseline[0])
    assert kept_by_one.all()


def test_the_mask_survives_padding_of_a_non_square_image(corpus):
    """The trap: resizing the mask without padding it first shifts the lungs by
    the padding offset, so the ablation blanks the wrong anatomy."""
    only, _ = _dataset(corpus, LUNGS_ONLY)[0]
    baseline, _ = _dataset(corpus, BASELINE)[0]

    # The bright band in the baseline and the kept band in lungs_only must be
    # the same rows. Compare their centres of mass.
    kept = np.isclose(only[0], baseline[0]) & (baseline[0] > baseline[0].mean())
    bright = baseline[0] > baseline[0].mean()
    assert kept.any() and bright.any()
    assert abs(np.argwhere(kept)[:, 0].mean() - np.argwhere(bright)[:, 0].mean()) < 1.5


def test_rows_without_a_mask_are_refused(corpus):
    root, frame = corpus
    frame = pd.concat([frame, pd.DataFrame([_row("c1", "images/c0.png", None)])])
    with pytest.raises(SplitError, match="must carry a mask"):
        AblationDataset(
            frame, spec=SPEC, variant=LUNGS_ONLY,
            mask_roots={"covid_radiography": root}, image_roots={"covid_radiography": root},
        )


def test_an_unknown_variant_is_refused(corpus):
    root, frame = corpus
    with pytest.raises(ValueError, match="unknown variant"):
        AblationDataset(
            frame, spec=SPEC, variant="everything",
            mask_roots={"covid_radiography": root}, image_roots={"covid_radiography": root},
        )
