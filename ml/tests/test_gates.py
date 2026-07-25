"""Every gate is tested twice: that it passes clean data, and that it fires.

The second half is the one that matters. A leakage detector which has only ever
been run on data believed to be clean has not been tested at all — it would
pass identically if it were a function returning True.
"""

from __future__ import annotations

import pandas as pd
import pytest
from cxr import manifest, splits
from cxr.gates import (
    GateStatus,
    class_source_independence,
    near_duplicate_disjointness,
    patient_disjointness,
    run_all,
    source_confound_probe,
)
from cxr.gates.g1_patient import external_is_source_pure
from cxr.gates.g4_class_source import cramers_v
from cxr.gates.runner import any_failed, render, to_json
from cxr.hashing import dhash, sha256_file
from cxr.splits import TEST, TRAIN, SplitConfig
from PIL import Image

# --------------------------------------------------------------------------
# G1 · patient disjointness
# --------------------------------------------------------------------------


def test_g1_passes_on_a_grouped_split(clean_corpus):
    _, frame = clean_corpus
    result = patient_disjointness(frame, splits.assign(frame))
    assert result.status is GateStatus.PASS
    assert result.measured == 0.0


def test_g1_catches_a_patient_moved_across_the_split(clean_corpus):
    """The classic mistake: splitting on images instead of patients, so a
    follow-up film of the same chest ends up in test."""
    _, frame = clean_corpus
    assigned = splits.assign(frame)

    train_patient = frame[assigned == TRAIN]["patient_id"].iloc[0]
    victim = frame.index[frame["patient_id"] == train_patient][0]
    assigned.loc[victim] = TEST

    result = patient_disjointness(frame, assigned)
    assert result.status is GateStatus.FAIL
    assert result.measured == 1.0
    assert train_patient in result.details["straddling_patients"]


def test_g1_ignores_splits_it_is_told_to(clean_corpus):
    _, frame = clean_corpus
    assigned = splits.assign(frame)
    victim = frame.index[0]
    assigned.loc[victim] = TEST
    patient = frame.loc[victim, "patient_id"]

    unfiltered = patient_disjointness(frame, assigned)
    filtered = patient_disjointness(frame, assigned, ignore=(TEST,))
    if unfiltered.status is GateStatus.FAIL:
        assert patient not in filtered.details.get("straddling_patients", [])


def test_g1b_rejects_an_external_split_that_shares_a_source(clean_corpus):
    _, frame = clean_corpus
    assigned = splits.assign(frame, SplitConfig(holdout_source="beta", calibration_folds=4))
    assert external_is_source_pure(frame, assigned).status is GateStatus.PASS

    leaked = assigned.copy()
    alpha_row = frame.index[frame["source"] == "alpha"][0]
    leaked.loc[alpha_row] = "external"
    assert external_is_source_pure(frame, leaked).status is GateStatus.FAIL


def test_g1b_skips_when_no_external_split_exists(clean_corpus):
    _, frame = clean_corpus
    result = external_is_source_pure(frame, splits.assign(frame))
    assert result.status is GateStatus.SKIPPED


# --------------------------------------------------------------------------
# G2 · near-duplicate disjointness
# --------------------------------------------------------------------------


def test_g2_passes_when_every_image_is_distinct(clean_corpus):
    _, frame = clean_corpus
    result = near_duplicate_disjointness(frame, splits.assign(frame))
    assert result.status is GateStatus.PASS


def test_g2_catches_a_recompressed_copy_in_another_split(clean_corpus):
    """The aggregate collections contain each other's images, re-encoded.

    Different bytes, different filename, no shared patient id — invisible to
    SHA-256 and to G1, and it puts the same chest on both sides of the split.
    """
    root, frame = clean_corpus
    assigned = splits.assign(frame)

    original_row = frame[assigned == TRAIN].iloc[0]
    with Image.open(root / original_row["path"]) as image:
        copy_path = root / "aggregate" / "reposted.jpg"
        copy_path.parent.mkdir(parents=True, exist_ok=True)
        image.save(copy_path, quality=75)
        copy_hash = dhash(Image.open(copy_path))

    duplicate = original_row.copy()
    duplicate["image_id"] = "aggregate:reposted.jpg"
    duplicate["source"] = "aggregate"
    duplicate["path"] = "aggregate/reposted.jpg"
    duplicate["patient_id"] = "aggregate:p9999"
    duplicate["sha256"] = sha256_file(copy_path)
    duplicate["phash"] = copy_hash

    extended = manifest.from_records([*frame.to_dict("records"), duplicate.to_dict()])
    extended_splits = pd.concat([assigned, pd.Series([TEST], index=[len(frame)])])

    result = near_duplicate_disjointness(extended, extended_splits)
    assert result.status is GateStatus.FAIL
    assert result.measured == 1.0


def test_g2_reports_exact_duplicates_even_when_they_pass(clean_corpus):
    _, frame = clean_corpus
    assigned = splits.assign(frame)
    result = near_duplicate_disjointness(frame, assigned)
    assert "exact_duplicate_images" in result.details


def test_g2_skips_without_hashes(clean_corpus):
    _, frame = clean_corpus
    frame = frame.copy()
    frame["phash"] = None
    result = near_duplicate_disjointness(frame, splits.assign(frame))
    assert result.status is GateStatus.SKIPPED


# --------------------------------------------------------------------------
# G3 · source-confound probe
# --------------------------------------------------------------------------


def test_g3_skips_with_a_single_source(clean_corpus):
    root, frame = clean_corpus
    single = frame[frame["source"] == "alpha"]
    assert source_confound_probe(single, image_root=root).status is GateStatus.SKIPPED


def test_g3_catches_visually_distinguishable_sources(confounded_corpus):
    """The diagnostic the field skipped.

    The COVID source here carries a white border and its own contrast curve,
    which is exactly the kind of acquisition signature that separates real
    repositories. A linear model on thumbnails should find it immediately.
    """
    root, frame = confounded_corpus
    result = source_confound_probe(frame, image_root=root)
    assert result.status is GateStatus.FAIL
    assert result.measured > 0.75
    assert result.details["chance_level"] == pytest.approx(0.5)


def test_g3_passes_when_sources_are_processed_identically(clean_corpus):
    root, frame = clean_corpus
    result = source_confound_probe(frame, image_root=root)
    assert result.status is GateStatus.PASS


def test_g3_accepts_precomputed_features(confounded_corpus):
    root, frame = confounded_corpus
    from cxr.gates.g3_source_probe import thumbnail_features

    features = thumbnail_features([root / path for path in frame["path"]])
    assert source_confound_probe(frame, features=features).status is GateStatus.FAIL


def test_g3_rejects_a_feature_matrix_of_the_wrong_length(confounded_corpus):
    root, frame = confounded_corpus
    from cxr.gates.g3_source_probe import thumbnail_features

    features = thumbnail_features([root / path for path in frame["path"]])[:-1]
    with pytest.raises(ValueError, match="manifest has"):
        source_confound_probe(frame, features=features)


# --------------------------------------------------------------------------
# G4 · class-source independence
# --------------------------------------------------------------------------


def test_cramers_v_is_zero_for_independent_variables():
    table = [[50.0, 50.0], [50.0, 50.0]]
    assert cramers_v(pd.DataFrame(table).to_numpy()) == pytest.approx(0.0, abs=1e-9)


def test_cramers_v_is_high_when_one_determines_the_other():
    table = [[100.0, 0.0], [0.0, 100.0]]
    assert cramers_v(pd.DataFrame(table).to_numpy()) > 0.9


def test_g4_passes_when_every_class_appears_in_every_source(clean_corpus):
    _, frame = clean_corpus
    result = class_source_independence(frame, splits.assign(frame))
    assert result.status is GateStatus.PASS
    assert result.details["classes_from_a_single_source"] == []


def test_g4_catches_a_class_drawn_from_one_source(confounded_corpus):
    """The structural problem with the brief's dataset list.

    ChestX-ray14 has no COVID label, so COVID has to come from somewhere else,
    and then recognising the source is sufficient to predict the class.
    """
    _, frame = confounded_corpus
    result = class_source_independence(frame)
    assert result.status is GateStatus.FAIL
    # Not just covid: in this configuration *every* class is single-sourced,
    # because the pre-pandemic set supplies normal and pneumonia and nothing
    # else, and the pandemic-era set supplies covid and nothing else.
    assert result.details["classes_from_a_single_source"] == ["covid", "normal", "pneumonia"]
    assert "covid" in result.summary


def test_g4_skips_with_a_single_source(clean_corpus):
    _, frame = clean_corpus
    single = frame[frame["source"] == "alpha"]
    assert class_source_independence(single).status is GateStatus.SKIPPED


def test_g4_excludes_the_external_split_from_per_split_scores(clean_corpus):
    """The external split is one source by construction; V is undefined there
    and its absence should not read as a finding."""
    _, frame = clean_corpus
    assigned = splits.assign(frame, SplitConfig(holdout_source="beta", calibration_folds=4))
    result = class_source_independence(frame, assigned)
    assert "external" not in result.details["per_split"]


# --------------------------------------------------------------------------
# Runner
# --------------------------------------------------------------------------


def test_runner_passes_everything_on_a_clean_corpus(clean_corpus):
    root, frame = clean_corpus
    results = run_all(frame, splits.assign(frame), image_root=root)
    assert not any_failed(results), render(results)


def test_runner_fails_on_the_confounded_corpus(confounded_corpus):
    root, frame = confounded_corpus
    assigned = splits.assign(frame, SplitConfig(n_folds=4, calibration_folds=4))
    results = run_all(frame, assigned, image_root=root)

    assert any_failed(results)
    failed = {result.gate for result in results if result.failed}
    assert {"G3", "G4"} <= failed


def test_runner_reports_every_gate_rather_than_stopping_at_the_first(confounded_corpus):
    root, frame = confounded_corpus
    assigned = splits.assign(frame, SplitConfig(n_folds=4, calibration_folds=4))
    results = run_all(frame, assigned, image_root=root)
    assert {result.gate for result in results} == {"G1", "G1b", "G2", "G3", "G4"}


def test_runner_skips_g3_without_pixels(clean_corpus):
    _, frame = clean_corpus
    results = run_all(frame, splits.assign(frame))
    probe = next(result for result in results if result.gate == "G3")
    assert probe.status is GateStatus.SKIPPED


def test_results_serialise_for_the_model_card(clean_corpus, tmp_path):
    root, frame = clean_corpus
    results = run_all(frame, splits.assign(frame), image_root=root)
    path = tmp_path / "reports" / "gates.json"
    to_json(results, path)

    import json

    payload = json.loads(path.read_text())
    assert {entry["gate"] for entry in payload} == {"G1", "G1b", "G2", "G3", "G4"}
    probe = next(entry for entry in payload if entry["gate"] == "G3")
    assert probe["measured"] is not None


def test_render_names_the_failures(confounded_corpus):
    root, frame = confounded_corpus
    assigned = splits.assign(frame, SplitConfig(n_folds=4, calibration_folds=4))
    text = render(run_all(frame, assigned, image_root=root))
    assert "gates failed" in text
    assert "Do not train" in text
