"""Duplicate grouping, and that splitting respects it."""

from __future__ import annotations

import pandas as pd
import pytest
from cxr import dedupe, splits
from cxr.manifest import Label, LabelProvenance, View


def _row(image_id, source, phash, label=Label.NORMAL, patient=None):
    return {
        "image_id": image_id,
        "source": source,
        "path": f"{image_id}.png",
        "label": str(label),
        "label_raw": str(label),
        "label_provenance": str(LabelProvenance.RADIOLOGIST),
        "patient_id": patient or f"{source}:{image_id}",
        "study_id": None,
        "view": str(View.PA),
        "age": 50.0,
        "sex": "F",
        "sha256": f"sha-{image_id}",
        "phash": phash,
        "width": 299,
        "height": 299,
        "mask_path": None,
    }


def _hash(bits: str) -> str:
    """A 256-bit hash written as its leading hex digits, zero-padded."""
    return bits.ljust(64, "0")


TWIN = _hash("abc123")
OTHER = _hash("f0f0f0")


@pytest.fixture
def twinned():
    """The real corpus in miniature: one image present in both collections."""
    return pd.DataFrame(
        [
            _row("a1", "rsna_pneumonia", TWIN),
            _row("b1", "covid_radiography", TWIN),
            _row("a2", "rsna_pneumonia", OTHER),
        ]
    )


def test_every_row_gets_a_group_including_singletons(twinned):
    annotated = dedupe.annotate(twinned)
    assert annotated[dedupe.GROUP_COLUMN].notna().all()
    assert annotated[dedupe.GROUP_COLUMN].nunique() == 2


def test_the_twins_share_a_group(twinned):
    annotated = dedupe.annotate(twinned)
    groups = annotated.set_index("image_id")[dedupe.GROUP_COLUMN]
    assert groups["a1"] == groups["b1"]
    assert groups["a2"] != groups["a1"]


def test_rows_without_a_hash_do_not_all_collide():
    """A shared NaN bucket would glue unrelated images into one group."""
    frame = pd.DataFrame(
        [
            _row("a1", "rsna_pneumonia", None),
            _row("a2", "rsna_pneumonia", None),
            _row("a3", "rsna_pneumonia", OTHER),
        ]
    )
    annotated = dedupe.annotate(frame)
    assert annotated[dedupe.GROUP_COLUMN].nunique() == 3


def test_deduplicate_keeps_the_preferred_source(twinned):
    kept, dropped = dedupe.deduplicate(twinned)
    assert sorted(kept["image_id"]) == ["a1", "a2"]
    assert dropped["image_id"].tolist() == ["b1"]


def test_preference_order_is_honoured(twinned):
    kept, _ = dedupe.deduplicate(twinned, preference=("covid_radiography", "rsna_pneumonia"))
    assert sorted(kept["image_id"]) == ["a2", "b1"]


def test_a_label_conflict_is_kept_whole_rather_than_resolved():
    """Discarding one side would bury the evidence that something is wrong."""
    frame = pd.DataFrame(
        [
            _row("a1", "rsna_pneumonia", TWIN, label=Label.NORMAL),
            _row("b1", "covid_radiography", TWIN, label=Label.PNEUMONIA),
        ]
    )
    kept, dropped = dedupe.deduplicate(frame)
    assert len(kept) == 2
    assert dropped.empty
    assert int(dedupe.summarise(frame)["label_conflicts"].iloc[0]) == 1


def test_summarise_counts_cross_source_clusters(twinned):
    report = dedupe.summarise(twinned)
    assert int(report["clusters"].iloc[0]) == 1
    assert int(report["images"].iloc[0]) == 2
    assert int(report["cross_source"].iloc[0]) == 1
    assert int(report["label_conflicts"].iloc[0]) == 0


# --------------------------------------------------------------------------
# Splitting has to see the groups
# --------------------------------------------------------------------------


def test_grouping_key_merges_patient_and_cluster():
    """Two patients joined by a shared image become one splitting unit."""
    frame = dedupe.annotate(
        pd.DataFrame(
            [
                _row("a1", "rsna_pneumonia", TWIN, patient="rsna:p1"),
                _row("a2", "rsna_pneumonia", OTHER, patient="rsna:p1"),
                _row("b1", "covid_radiography", TWIN, patient="covid:p9"),
            ]
        )
    )
    keys = splits.grouping_key(frame)
    assert keys.nunique() == 1


def test_grouping_key_falls_back_to_patient_without_annotation(twinned):
    keys = splits.grouping_key(twinned)
    assert keys.tolist() == twinned["patient_id"].tolist()


def test_twins_never_straddle_a_split():
    """The failure G2 found on the real corpus, reproduced and then fixed."""
    import random

    # Random 256-bit signatures sit ~128 bits apart, so each pair forms its own
    # cluster rather than chaining into one -- which is what distinct chests
    # look like, and what the sweep on the real corpus measured.
    rng = random.Random(0)
    rows = []
    for index in range(60):
        digest = f"{rng.getrandbits(256):064x}"
        label = [Label.NORMAL, Label.PNEUMONIA, Label.COVID][index % 3]
        rows.append(_row(f"r{index}", "rsna_pneumonia", digest, label=label))
        rows.append(_row(f"c{index}", "covid_radiography", digest, label=label))
    frame = dedupe.annotate(pd.DataFrame(rows))

    assigned = splits.assign(frame, splits.SplitConfig(n_folds=3, calibration_folds=3))
    spans = frame.assign(split=assigned).groupby(dedupe.GROUP_COLUMN)["split"].nunique()
    assert int(spans.max()) == 1
