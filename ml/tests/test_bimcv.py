"""BIMCV ingest: projection filtering, and two definitions of the label."""

from __future__ import annotations

import numpy as np
import pytest
from cxr.manifest import Label, View
from cxr.sources import SourceError, get
from PIL import Image


def _image(root, partition, subject, session, filename):
    path = root / partition / subject / session / "mod-rx" / filename
    path.parent.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(abs(hash(filename)) % 2**32)
    Image.fromarray((rng.random((32, 32)) * 255).astype(np.uint8), mode="L").save(path)
    return path


def _scans(root, partition, subject, session, rows):
    path = root / partition / subject / session / f"{subject}_{session}_scans.tsv"
    path.parent.mkdir(parents=True, exist_ok=True)
    header = [
        "filename", "Patient's Sex (00100040)", "Patient's Age (00101010)",
        "Series Description (0008103E)", "Manufacturer (00080070)",
    ]
    lines = ["\t".join(header)]
    lines += ["\t".join(row) for row in rows]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


@pytest.fixture
def corpus(tmp_path):
    root = tmp_path / "bimcv"
    for partition, subject, session in [
        ("covid19_posi", "sub-S0001", "ses-E0001"),
        ("covid19_neg", "sub-S0002", "ses-E0002"),
    ]:
        stem = f"{subject}_{session}_run-1_bp-chest"
        _image(root, partition, subject, session, f"{stem}_vp-pa_cr.png")
        _image(root, partition, subject, session, f"{stem}_vp-ap_dx.png")
        _image(root, partition, subject, session, f"{stem}_vp-ll_cr.png")
        # The trap: a lateral with no vp- marker, identifiable only from the
        # Spanish series description.
        _image(root, partition, subject, session, f"{stem}_cr.png")
        # The two partitions write the `filename` column differently: the
        # positives prefix it with the modality directory, the negatives do
        # not. Both forms appear here because a fixture that used only one of
        # them let a join failure ship -- the adapter matched every negative
        # and no positive, and nothing failed loudly enough to notice.
        key = f"mod-rx/{stem}" if partition == "covid19_posi" else stem
        _scans(root, partition, subject, session, [
            [f"{key}_vp-pa_cr.png", "F", "064Y", "W033 Torax PA", "SIEMENS"],
            [f"{key}_vp-ap_dx.png", "F", "064Y", "W031 Torax AP", "SIEMENS"],
            [f"{key}_vp-ll_cr.png", "F", "064Y", "W034 Torax Lat.", "SIEMENS"],
            [f"{key}_cr.png", "F", "064Y", "W034 Torax Lat.", "SIEMENS"],
        ])
    return root


def test_only_frontal_projections_are_admitted(corpus):
    records = get("bimcv_covid19").build(corpus)
    assert len(records) == 4  # pa + ap, per partition
    assert {r["view"] for r in records} == {str(View.PA), str(View.AP)}


def test_a_lateral_without_a_view_marker_is_still_excluded(corpus):
    """Filtering on vp-ll alone admits these as unknown-view, and a lateral is
    a different projection rather than a hard frontal case."""
    paths = [r["path"] for r in get("bimcv_covid19").build(corpus)]
    assert not any(p.endswith("bp-chest_cr.png") for p in paths)
    assert not any("vp-ll" in p for p in paths)


def test_negatives_are_non_covid_never_normal(corpus):
    """Only a fifth of the negative partition is radiologically clear; calling
    them normal turns Track 2 into sick-versus-healthy."""
    records = get("bimcv_covid19").build(corpus)
    labels = {r["label"] for r in records}
    assert labels == {str(Label.COVID), str(Label.NON_COVID)}
    assert str(Label.NORMAL) not in labels


def test_the_label_is_molecular_status_not_the_series_description(corpus):
    """label_raw must not look like a finding when it carries a protocol name."""
    for record in get("bimcv_covid19").build(corpus):
        assert record["label_raw"] in {"covid19_posi", "covid19_neg"}
        assert "Torax" not in record["label_raw"]
        assert record["label_provenance"] == "pcr"


def test_demographics_come_from_the_session_metadata(corpus):
    """Every record, not the first one, and both partitions.

    Checking one record hid a join that worked for the negatives and failed for
    the positives. That failure mode is worse than missing demographics: it
    makes "sex is known" a perfect predictor of the label, so anything
    downstream that touches demographics leaks the answer.
    """
    records = get("bimcv_covid19").build(corpus)
    by_label = {record["label"] for record in records}
    assert by_label == {"covid", "non_covid"}
    for record in records:
        assert record["age"] == 64.0, record["image_id"]
        assert record["sex"] == "F", record["image_id"]


def test_patient_ids_are_namespaced_and_sessions_kept(corpus):
    for record in get("bimcv_covid19").build(corpus):
        assert record["patient_id"].startswith("bimcv_covid19:sub-S")
        assert record["study_id"].startswith("ses-E")


def test_a_directory_with_no_partition_is_rejected(tmp_path):
    with pytest.raises(SourceError, match="no BIMCV partition"):
        get("bimcv_covid19").build(tmp_path)


def test_a_partition_with_only_laterals_is_an_error_not_an_empty_manifest(tmp_path):
    root = tmp_path / "bimcv"
    stem = "sub-S0003_ses-E0003_run-1_bp-chest"
    _image(root, "covid19_neg", "sub-S0003", "ses-E0003", f"{stem}_vp-ll_cr.png")
    with pytest.raises(SourceError, match="no frontal chest radiographs"):
        get("bimcv_covid19").build(root)


def test_bimcv_is_a_second_covid_capable_source(corpus):
    """G4 fails by construction while only one collection can supply COVID.
    This is the source that changes that."""
    from cxr import sources

    assert "bimcv_covid19" in sources.covid_capable()
    assert len(sources.covid_capable()) >= 2
