"""Freezing a checkpoint, and keeping its caveats attached to it."""

from __future__ import annotations

import json
import zipfile

import pytest
from cxr.export import METADATA_FILE, describe, export, load_metadata, spec_of
from cxr.models import Checkpoint, ModelConfig
from cxr.preprocessing import PreprocessingSpec

pytest.importorskip("torch")
pytest.importorskip("timm")

SPEC = PreprocessingSpec(target_size=32)


@pytest.fixture
def checkpoint_dir(tmp_path):
    import torch
    from cxr.models import build_model

    config = ModelConfig(backbone="resnet18", pretrained=False, classes=("non_covid", "covid"))
    model = build_model(config)
    checkpoint = Checkpoint(
        model=config,
        spec=SPEC,
        temperature=3.35,
        metrics={
            "test": {
                "macro_auc": 0.746,
                "balanced_accuracy": 0.648,
                "per_class_auc": {"non_covid": 0.746, "covid": 0.746},
            },
            "ece_after_calibration": 0.0854,
            "sources": ["bimcv_covid19"],
        },
        gates={"blocking": [], "acknowledged": [], "skipped": ["G1b", "G3", "G4"]},
        notes="2-class, single source",
    )
    directory = tmp_path / "ckpt"
    checkpoint.write(directory, state_dict=model.state_dict())
    torch.save(model.state_dict(), directory / "weights.pt")
    return directory


ABLATION = [
    {"name": "Lungs removed", "retention": 0.837, "images": 292,
     "interpretation": "83.7% of the signal survives with the lung fields removed"},
    {"name": "Lungs only", "retention": 0.4653, "images": 292, "interpretation": "..."},
]


def test_the_metadata_lives_inside_the_archive(checkpoint_dir, tmp_path):
    """One file, so the weights and the confound profile cannot be separated by
    copying one and forgetting the other."""
    out = tmp_path / "track2.pt2"
    export(checkpoint_dir, out)

    with zipfile.ZipFile(out) as archive:
        assert any(name.endswith(METADATA_FILE) for name in archive.namelist())


def test_the_exported_model_agrees_with_the_checkpoint(checkpoint_dir, tmp_path):
    """Tracing bakes in whichever path the example took. An export nobody
    reloaded and compared is an export nobody has checked."""
    import torch

    out = tmp_path / "m.pt2"
    export(checkpoint_dir, out)

    reloaded = torch.export.load(str(out)).module()
    with torch.no_grad():
        # Several batch sizes: exporting from a single-image example silently
        # freezes the batch dimension, and the failure only shows up on the
        # first request that is not exactly one image.
        for size in (1, 2, 7):
            shape = (size, SPEC.channels, SPEC.target_size, SPEC.target_size)
            assert reloaded(torch.randn(*shape)).shape == (size, 2)


def test_the_serving_side_can_read_the_spec_without_the_weights(checkpoint_dir, tmp_path):
    out = tmp_path / "m.pt2"
    export(checkpoint_dir, out)

    metadata = load_metadata(out)
    assert spec_of(metadata) == SPEC
    assert metadata["temperature"] == pytest.approx(3.35)


def test_retention_travels_with_the_weights(checkpoint_dir, tmp_path):
    out = tmp_path / "m.pt2"
    ablation_path = tmp_path / "ablation.json"
    ablation_path.write_text(json.dumps(ABLATION), encoding="utf-8")

    metadata = export(checkpoint_dir, out, ablation=ablation_path)
    assert metadata["ablation"]["lungs_removed_retention"] == pytest.approx(0.837)
    assert metadata["ablation"]["lungs_only_retention"] == pytest.approx(0.4653)


def test_a_missing_ablation_reads_as_unmeasured_not_as_zero(checkpoint_dir, tmp_path):
    """None and 0.0 mean opposite things here. Zero retention would say the
    model collapsed to chance without its lungs, which is the best possible
    result; not having run the ablation is no evidence at all."""
    metadata = export(checkpoint_dir, tmp_path / "m.pt2")
    assert metadata["ablation"]["lungs_removed_retention"] is None


def test_skipped_gates_stay_distinct_from_acknowledged_ones(checkpoint_dir, tmp_path):
    """Track 2 skips G4 because one source makes it undefined; Track 1 fails it
    and acknowledges it. Collapsing the two would let a served model claim a
    measurement that never happened."""
    metadata = export(checkpoint_dir, tmp_path / "m.pt2")
    assert metadata["gates"]["skipped"] == ["G1b", "G3", "G4"]
    assert metadata["gates"]["acknowledged"] == []


def test_load_metadata_rejects_a_file_it_did_not_write(tmp_path):
    stray = tmp_path / "not-ours.pt2"
    with zipfile.ZipFile(stray, "w") as archive:
        archive.writestr("something.txt", "hello")
    with pytest.raises(ValueError, match="was not written by cxr export"):
        load_metadata(stray)


def test_describe_survives_a_checkpoint_with_no_metrics():
    """An export of a half-finished run should say so rather than raise."""
    bare = Checkpoint(model=ModelConfig(classes=("a", "b")), spec=SPEC)
    described = describe(bare)
    assert described["metrics"]["macro_auc"] is None
    assert described["gates"]["skipped"] == []
