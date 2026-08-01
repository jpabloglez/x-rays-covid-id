"""The training loop, end to end on a corpus small enough to run in CI."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from cxr.manifest import Label, LabelProvenance, View
from cxr.preprocessing import PreprocessingSpec
from PIL import Image

pytest.importorskip("torch")
pytest.importorskip("timm")
pytest.importorskip("monai")

SPEC = PreprocessingSpec(target_size=32)
LABELS = (Label.NORMAL, Label.PNEUMONIA, Label.COVID)
# Enough rows per split that every class appears in each, which the evaluation
# needs before it can report a per-class AUC at all.
SPLIT_SIZES = {"train": 24, "val": 12, "calib": 12, "test": 12}


def _row(image_id, path, label, split, source):
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
        "phash": "0" * 64,
        "width": 48,
        "height": 48,
        "mask_path": None,
        "split": split,
    }


@pytest.fixture
def corpus(tmp_path):
    """A learnable toy task: class index sets the brightness of a blob."""
    root = tmp_path / "images"
    (root / "images").mkdir(parents=True)
    rng = np.random.default_rng(0)
    rows, index = [], 0
    for split, count in SPLIT_SIZES.items():
        for position in range(count):
            label_index = position % 3
            pixels = rng.normal(60 + 60 * label_index, 12, (48, 48)).clip(0, 255)
            name = f"chest{index}.png"
            Image.fromarray(pixels.astype(np.uint8), mode="L").save(root / "images" / name)
            rows.append(
                _row(
                    f"c{index}",
                    f"images/{name}",
                    LABELS[label_index],
                    split,
                    "covid_radiography" if index % 2 else "rsna_pneumonia",
                )
            )
            index += 1
    return root, pd.DataFrame(rows)


def _train(corpus, tmp_path, config=None, **overrides):
    from cxr.models import ModelConfig
    from cxr.train import TrainConfig, train

    root, frame = corpus
    return train(
        frame,
        spec=SPEC,
        output=tmp_path / "model",
        image_roots={"covid_radiography": root, "rsna_pneumonia": root},
        model_config=ModelConfig(backbone="resnet18", pretrained=False),
        config=config
        or TrainConfig(epochs=3, batch_size=4, accumulate=1, workers=0, freeze_backbone_epochs=1),
        **overrides,
    )


def test_training_runs_and_writes_a_usable_checkpoint(corpus, tmp_path):
    from cxr.models import Checkpoint

    _train(corpus, tmp_path)
    assert (tmp_path / "model" / "weights.pt").exists()
    assert (tmp_path / "model" / "preprocessing.json").exists()

    reloaded = Checkpoint.read(tmp_path / "model")
    assert reloaded.spec == SPEC
    assert reloaded.model.classes == ("normal", "pneumonia", "covid")
    assert reloaded.temperature > 0


def test_the_selected_epoch_is_never_a_frozen_one(corpus, tmp_path):
    """A frozen epoch measures the head against a trunk that has not adapted."""
    checkpoint = _train(corpus, tmp_path)
    assert checkpoint.metrics["selected_epoch"] >= 1


def test_test_predictions_are_saved_for_later_analysis(corpus, tmp_path):
    """The confound battery needs per-image outputs, not just the summary."""
    _train(corpus, tmp_path)
    saved = np.load(tmp_path / "model" / "test_predictions.npz")
    assert saved["logits"].shape == (SPLIT_SIZES["test"], 3)
    assert len(saved["image_ids"]) == SPLIT_SIZES["test"]


def test_the_gate_report_travels_with_the_checkpoint(corpus, tmp_path):
    """A score quoted without its confound profile is the score Roberts et al.
    reviewed 415 of."""
    from cxr.train import report

    gates = {"acknowledged": ["G3", "G4"], "blocking": [], "training_permitted": True}
    checkpoint = _train(corpus, tmp_path, gates=gates)

    assert checkpoint.gates["acknowledged"] == ["G3", "G4"]
    text = report(checkpoint)
    assert "acknowledged confounds" in text
    assert "upper bound" in text


def test_a_report_without_acknowledgements_makes_no_confound_claim(corpus, tmp_path):
    from cxr.train import report

    assert "acknowledged confounds" not in report(_train(corpus, tmp_path))


def test_effective_batch_is_the_accumulated_one():
    from cxr.train import TrainConfig

    assert TrainConfig(batch_size=8, accumulate=2).effective_batch == 16


def test_an_unreachable_min_delta_stops_the_run_on_the_plateau(corpus, tmp_path):
    """Patience should measure progress, not any movement at all.

    With a gain of 2.0 required, no epoch can ever count as progress, so the
    run must stop as soon as `patience` epochs have passed rather than using
    its full budget chasing the fourth decimal place.
    """
    from cxr.train import TrainConfig

    checkpoint = _train(
        corpus,
        tmp_path,
        config=TrainConfig(
            epochs=12, batch_size=4, accumulate=1, workers=0,
            freeze_backbone_epochs=1, patience=1, min_delta=2.0,
        ),
    )
    assert len(checkpoint.metrics["history"]) < 12


def test_the_best_epoch_is_kept_even_when_it_did_not_count_as_progress(corpus, tmp_path):
    """min_delta governs whether to continue, never which weights to keep.

    A marginal winner found while the run is already on the plateau is still
    the best model seen, and discarding it would make early stopping cost
    accuracy rather than just time.
    """
    from cxr.train import TrainConfig

    checkpoint = _train(
        corpus,
        tmp_path,
        config=TrainConfig(
            epochs=4, batch_size=4, accumulate=1, workers=0,
            freeze_backbone_epochs=1, patience=1, min_delta=2.0,
        ),
    )
    history = checkpoint.metrics["history"]
    unfrozen = [entry["val_macro_auc"] for entry in history if not entry["backbone_frozen"]]
    assert checkpoint.metrics["val_macro_auc"] == pytest.approx(max(unfrozen))


def test_the_provenance_note_is_derived_from_the_corpus():
    """This string was hardcoded to "Track 1: pooled corpus, knowingly
    confounded" and printed on every run regardless of what was trained. It was
    true of everything that existed when it was written and false as soon as a
    second task did."""
    from cxr.data import CLASSES, TRACK2_CLASSES
    from cxr.train import _provenance_note

    pooled = _provenance_note(
        pd.DataFrame({"source": ["rsna_pneumonia", "covid_radiography"]}), CLASSES
    )
    assert "pooled across 2 sources" in pooled
    assert "may be confounded with the collection" in pooled

    single = _provenance_note(pd.DataFrame({"source": ["bimcv_covid19"] * 3}), TRACK2_CLASSES)
    assert "single source (bimcv_covid19)" in single
    assert "by construction, not by measurement" in single
    assert "2-class (non_covid, covid)" in single


def test_the_single_source_note_does_not_claim_freedom_from_every_confound():
    """One source rules out the class-source confound and nothing else. Track 2
    still carries a view-position and a sex skew."""
    from cxr.data import TRACK2_CLASSES
    from cxr.train import _provenance_note

    note = _provenance_note(pd.DataFrame({"source": ["bimcv_covid19"]}), TRACK2_CLASSES)
    assert "Confounds within the source remain possible" in note
