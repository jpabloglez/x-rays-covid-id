"""Splitting rules: grouped by patient, and one source held out entirely."""

from __future__ import annotations

import pytest
from cxr import splits
from cxr.splits import CALIB, EXTERNAL, TEST, TRAIN, VAL, SplitConfig, SplitError


def test_every_row_gets_exactly_one_split(clean_corpus):
    _, frame = clean_corpus
    assigned = splits.assign(frame)
    assert assigned.notna().all()
    assert set(assigned) <= set(splits.SPLIT_NAMES)


def test_no_patient_straddles_a_split(clean_corpus):
    _, frame = clean_corpus
    assigned = splits.assign(frame)
    per_patient = frame.assign(split=assigned).groupby("patient_id")["split"].nunique()
    assert per_patient.max() == 1


def test_the_held_out_source_never_appears_elsewhere(clean_corpus):
    _, frame = clean_corpus
    assigned = splits.assign(frame, SplitConfig(holdout_source="beta", calibration_folds=4))
    assert set(frame[assigned == EXTERNAL]["source"]) == {"beta"}
    assert "beta" not in set(frame[assigned != EXTERNAL]["source"])


def test_calibration_is_disjoint_from_validation(clean_corpus):
    """Temperature scaling fitted on the early-stopping set is fitted on data
    the model was already selected against, which defeats the point."""
    _, frame = clean_corpus
    assigned = splits.assign(frame)
    calibration = set(frame[assigned == CALIB]["patient_id"])
    validation = set(frame[assigned == VAL]["patient_id"])
    assert calibration
    assert not calibration & validation


def test_all_five_splits_are_populated(clean_corpus):
    _, frame = clean_corpus
    assigned = splits.assign(frame, SplitConfig(holdout_source="beta", calibration_folds=4))
    assert set(assigned) == {TRAIN, VAL, CALIB, TEST, EXTERNAL}


def test_splitting_is_deterministic_for_a_seed(clean_corpus):
    _, frame = clean_corpus
    first = splits.assign(frame, SplitConfig(seed=7))
    second = splits.assign(frame, SplitConfig(seed=7))
    assert first.equals(second)


def test_a_different_seed_gives_a_different_split(clean_corpus):
    _, frame = clean_corpus
    assert not splits.assign(frame, SplitConfig(seed=1)).equals(
        splits.assign(frame, SplitConfig(seed=2))
    )


def test_unknown_holdout_source_is_rejected(clean_corpus):
    _, frame = clean_corpus
    with pytest.raises(SplitError, match="not present in the manifest"):
        splits.assign(frame, SplitConfig(holdout_source="gamma"))


def test_too_many_folds_for_the_data_is_rejected(clean_corpus):
    _, frame = clean_corpus
    with pytest.raises(SplitError, match="cannot make"):
        splits.assign(frame, SplitConfig(n_folds=500))


def test_leave_one_source_out_covers_every_source(clean_corpus):
    _, frame = clean_corpus
    folds = list(splits.leave_one_source_out(frame))
    assert [source for source, _, _ in folds] == ["alpha", "beta"]
    for source, train_index, test_index in folds:
        assert set(frame.loc[test_index, "source"]) == {source}
        assert source not in set(frame.loc[train_index, "source"])
        assert not set(train_index) & set(test_index)


def test_summary_reports_patients_and_sources(clean_corpus):
    _, frame = clean_corpus
    assigned = splits.assign(frame, SplitConfig(holdout_source="beta", calibration_folds=4))
    summary = splits.summarise(frame, assigned)
    assert summary.loc[EXTERNAL, "sources"] == 1
    assert summary["images"].sum() == len(frame)
