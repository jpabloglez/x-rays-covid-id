"""The manifest's invariants are load-bearing; these check they actually bear."""

from __future__ import annotations

import pandas as pd
import pytest
from cxr import manifest
from cxr.manifest import ManifestError


def test_a_clean_corpus_validates(clean_corpus):
    _, frame = clean_corpus
    assert len(manifest.validate(frame)) == len(frame)


def test_duplicate_image_ids_are_rejected(clean_corpus):
    _, frame = clean_corpus
    doubled = pd.concat([frame, frame.head(1)], ignore_index=True)
    with pytest.raises(ManifestError, match="duplicate values"):
        manifest.validate(doubled)


def test_patient_ids_must_be_namespaced_by_source(clean_corpus):
    """Patient 1 in RSNA and patient 1 in ChestX-ray14 are different people.

    Without the prefix they collide, and G1 then reports a disjointness it has
    not actually verified.
    """
    _, frame = clean_corpus
    frame = frame.copy()
    frame.loc[frame.index[0], "patient_id"] = "p0000"
    with pytest.raises(ManifestError, match="not prefixed with their source"):
        manifest.validate(frame)


def test_unknown_label_is_rejected(clean_corpus):
    _, frame = clean_corpus
    frame = frame.copy()
    frame.loc[frame.index[0], "label"] = "COVID-19"
    with pytest.raises(ManifestError, match="unexpected values"):
        manifest.validate(frame)


def test_missing_required_field_is_rejected(clean_corpus):
    _, frame = clean_corpus
    frame = frame.copy()
    frame.loc[frame.index[0], "sha256"] = None
    with pytest.raises(ManifestError, match="sha256"):
        manifest.validate(frame)


def test_all_problems_are_reported_together(clean_corpus):
    _, frame = clean_corpus
    frame = frame.copy()
    frame.loc[frame.index[0], "label"] = "bogus"
    frame.loc[frame.index[1], "view"] = "oblique"
    with pytest.raises(ManifestError) as caught:
        manifest.validate(frame)
    message = str(caught.value)
    assert "label" in message
    assert "view" in message


def test_age_outside_a_human_range_is_rejected(clean_corpus):
    _, frame = clean_corpus
    frame = frame.copy()
    frame.loc[frame.index[0], "age"] = 900.0
    with pytest.raises(ManifestError, match="age"):
        manifest.validate(frame)


def test_optional_fields_may_be_null(clean_corpus):
    _, frame = clean_corpus
    frame = frame.copy()
    frame["age"] = None
    frame["sex"] = None
    frame["study_id"] = None
    assert len(manifest.validate(frame)) == len(frame)


def test_roundtrip_through_parquet(clean_corpus, tmp_path):
    _, frame = clean_corpus
    path = tmp_path / "manifests" / "corpus.parquet"
    manifest.write(frame, path)
    assert manifest.read(path).equals(frame)


def test_summary_shows_a_class_confined_to_one_source(confounded_corpus):
    """The table to read before anything else: covid appears under one source."""
    _, frame = confounded_corpus
    summary = manifest.summarise(frame)
    assert summary.loc["chestxray14", "covid"] == 0
    assert summary.loc["covid_radiography", "covid"] > 0
    assert summary.loc["covid_radiography", "normal"] == 0
