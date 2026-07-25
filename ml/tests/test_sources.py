"""Adapter tests against synthetic directory layouts.

The real downloads are tens of gigabytes behind registration walls, so what is
tested here is the parsing and — more importantly — the label mapping, which is
where the damaging mistakes live. Calling an abnormal chest "normal" because it
happens to lack the one finding you care about is a data bug that no amount of
later modelling recovers from.
"""

from __future__ import annotations

import numpy as np
import pytest
from cxr import manifest, sources
from cxr.manifest import Label, View
from cxr.sources import ChestXray14, CovidRadiography, SourceError
from PIL import Image


def _png(path, seed=0):
    path.parent.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    Image.fromarray((rng.random((32, 32)) * 255).astype(np.uint8), mode="L").save(path)


# --------------------------------------------------------------------------
# Registry
# --------------------------------------------------------------------------


def test_only_one_registered_source_can_supply_covid():
    """The structural finding, asserted so it cannot be forgotten.

    ChestX-ray14 and RSNA predate the pandemic. If this test ever fails
    because a second COVID-capable source was added, that is good news and the
    confound analysis should be revisited.
    """
    assert sources.covid_capable() == ["covid_radiography"]


def test_unknown_source_is_rejected():
    with pytest.raises(SourceError, match="unknown source"):
        sources.get("chexpert")


def test_every_registered_source_declares_its_limits():
    for name, source in sources.REGISTRY.items():
        assert source.info.name == name
        assert source.info.citation
        assert source.info.licence


# --------------------------------------------------------------------------
# COVID-19 Radiography Database
# --------------------------------------------------------------------------


def test_covid_radiography_maps_folders_to_labels(tmp_path):
    for folder, count in [("COVID", 2), ("Normal", 2), ("Viral Pneumonia", 1)]:
        for index in range(count):
            _png(tmp_path / folder / "images" / f"{folder}-{index}.png", seed=index)

    frame = manifest.from_records(CovidRadiography().build(tmp_path))
    manifest.validate(frame)

    assert set(frame["label"]) == {Label.COVID, Label.NORMAL, Label.PNEUMONIA}
    assert len(frame) == 5


def test_covid_radiography_excludes_lung_opacity_by_default(tmp_path):
    """Neither confirmed pneumonia nor normal. Folding it in silently would
    change what the confusion matrix means."""
    _png(tmp_path / "COVID" / "images" / "c0.png")
    _png(tmp_path / "Lung_Opacity" / "images" / "l0.png", seed=1)

    default = manifest.from_records(CovidRadiography().build(tmp_path))
    assert len(default) == 1

    opted_in = manifest.from_records(
        CovidRadiography(include_lung_opacity=True).build(tmp_path)
    )
    assert len(opted_in) == 2
    assert "Lung_Opacity" in set(opted_in["label_raw"])


def test_covid_radiography_records_masks_when_present(tmp_path):
    _png(tmp_path / "COVID" / "images" / "c0.png")
    _png(tmp_path / "COVID" / "masks" / "c0.png", seed=9)
    _png(tmp_path / "Normal" / "images" / "n0.png", seed=2)

    frame = manifest.from_records(CovidRadiography().build(tmp_path))
    with_mask = frame[frame["path"].str.contains("COVID")]
    without = frame[frame["path"].str.contains("Normal")]
    assert with_mask["mask_path"].notna().all()
    assert without["mask_path"].isna().all()


def test_covid_radiography_view_is_unknown_not_assumed(tmp_path):
    """This dataset does not record projection. Assuming PA would corrupt the
    view stratification everywhere it is joined."""
    _png(tmp_path / "COVID" / "images" / "c0.png")
    frame = manifest.from_records(CovidRadiography().build(tmp_path))
    assert set(frame["view"]) == {View.UNKNOWN}


def test_covid_radiography_patient_ids_are_synthetic_and_namespaced(tmp_path):
    _png(tmp_path / "COVID" / "images" / "c0.png")
    _png(tmp_path / "COVID" / "images" / "c1.png", seed=1)
    frame = manifest.from_records(CovidRadiography().build(tmp_path))
    assert frame["patient_id"].nunique() == 2
    assert frame["patient_id"].str.startswith("covid_radiography:").all()


def test_covid_radiography_rejects_an_unexpected_layout(tmp_path):
    (tmp_path / "somethingelse").mkdir()
    with pytest.raises(SourceError, match="expected the"):
        CovidRadiography().build(tmp_path)


# --------------------------------------------------------------------------
# NIH ChestX-ray14
# --------------------------------------------------------------------------


def _nih_corpus(tmp_path, rows):
    import pandas as pd

    for index, row in enumerate(rows):
        _png(tmp_path / "images" / row["Image Index"], seed=index)
    pd.DataFrame(rows).to_csv(tmp_path / "Data_Entry_2017.csv", index=False)


def test_nih_maps_pneumonia_and_no_finding_only(tmp_path):
    """The critical mapping: a chest with emphysema is not a normal chest."""
    _nih_corpus(
        tmp_path,
        [
            _nih_row("a.png", "Pneumonia", 1),
            _nih_row("b.png", "No Finding", 2),
            _nih_row("c.png", "Emphysema|Cardiomegaly", 3),
            _nih_row("d.png", "Pneumonia|Effusion", 4),
        ],
    )
    frame = manifest.from_records(ChestXray14().build(tmp_path))
    manifest.validate(frame)

    by_file = dict(zip(frame["image_id"], frame["label"], strict=True))
    assert by_file["chestxray14:a.png"] == Label.PNEUMONIA
    assert by_file["chestxray14:b.png"] == Label.NORMAL
    assert by_file["chestxray14:d.png"] == Label.PNEUMONIA
    assert "chestxray14:c.png" not in by_file


def test_nih_keeps_the_raw_label_for_audit(tmp_path):
    _nih_corpus(tmp_path, [_nih_row("a.png", "Pneumonia|Effusion", 1)])
    frame = manifest.from_records(ChestXray14().build(tmp_path))
    assert frame["label_raw"].iloc[0] == "Pneumonia|Effusion"


def test_nih_carries_demographics_and_view(tmp_path):
    _nih_corpus(tmp_path, [_nih_row("a.png", "Pneumonia", 1, age=61, sex="F", view="AP")])
    frame = manifest.from_records(ChestXray14().build(tmp_path))
    assert frame["age"].iloc[0] == 61
    assert frame["sex"].iloc[0] == "F"
    assert frame["view"].iloc[0] == View.AP


def test_nih_drops_impossible_ages(tmp_path):
    """ChestX-ray14 genuinely contains ages above 400."""
    _nih_corpus(tmp_path, [_nih_row("a.png", "Pneumonia", 1, age=414)])
    frame = manifest.from_records(ChestXray14().build(tmp_path))
    assert frame["age"].isna().all()
    manifest.validate(frame)


def test_nih_groups_follow_ups_under_one_patient(tmp_path):
    _nih_corpus(
        tmp_path,
        [_nih_row("a.png", "Pneumonia", 7), _nih_row("b.png", "No Finding", 7)],
    )
    frame = manifest.from_records(ChestXray14().build(tmp_path))
    assert frame["patient_id"].nunique() == 1


def test_nih_requires_its_metadata_file(tmp_path):
    with pytest.raises(SourceError, match="Data_Entry_2017.csv not found"):
        ChestXray14().build(tmp_path)


def test_nih_rejects_metadata_missing_columns(tmp_path):
    import pandas as pd

    _png(tmp_path / "images" / "a.png")
    pd.DataFrame([{"Image Index": "a.png"}]).to_csv(
        tmp_path / "Data_Entry_2017.csv", index=False
    )
    with pytest.raises(SourceError, match="missing columns"):
        ChestXray14().build(tmp_path)


def _nih_row(filename, findings, patient, age=50, sex="M", view="PA"):
    return {
        "Image Index": filename,
        "Finding Labels": findings,
        "Patient ID": patient,
        "Patient Age": age,
        "Patient Gender": sex,
        "View Position": view,
    }
