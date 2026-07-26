"""Splitting, grouped by patient and duplicate cluster, and aware of source.

Three rules drive everything here. A patient's images never straddle a split,
or the test set measures memorisation. Neither do near-duplicates, which is a
separate rule because the same radiograph appears in several public
collections under unrelated patient ids — grouping by patient alone does not
see it. And one whole source stays out of training entirely, because the
number that matters is how the model does on a hospital it has never seen —
not on a held-out slice of the same pooled distribution.
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass

import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

from cxr.dedupe import GROUP_COLUMN as DUPLICATE_GROUP

TRAIN = "train"
VAL = "val"
CALIB = "calib"
TEST = "test"
EXTERNAL = "external"

SPLIT_NAMES = (TRAIN, VAL, CALIB, TEST, EXTERNAL)


class SplitError(ValueError):
    """Raised when a split cannot be produced as specified."""


@dataclass(frozen=True)
class SplitConfig:
    """How to carve the manifest.

    `holdout_source` is the external validation set and never appears in
    training in any fold. `calibration_folds` controls the size of the
    temperature-scaling slice: 10 folds means roughly a tenth of the training
    pool. That slice must be distinct from `val`, because a threshold fitted on
    the same data used for early stopping is fitted on data the model has
    already been selected against.
    """

    holdout_source: str | None = None
    n_folds: int = 5
    calibration_folds: int = 10
    seed: int = 0


def assign(frame: pd.DataFrame, config: SplitConfig | None = None) -> pd.Series:
    """Return a split label per row, aligned to `frame.index`."""
    config = config or SplitConfig()
    splits = pd.Series(index=frame.index, dtype="string")

    if config.holdout_source is not None:
        if config.holdout_source not in set(frame["source"]):
            raise SplitError(
                f"holdout_source {config.holdout_source!r} is not present in the manifest; "
                f"sources are {sorted(set(frame['source']))}"
            )
        external = frame["source"] == config.holdout_source
        splits[external] = EXTERNAL
        development = frame[~external]
    else:
        development = frame

    if development.empty:
        raise SplitError("no rows left for development after holding out the external source")

    folds = _grouped_folds(
        development, n_splits=config.n_folds, seed=config.seed, parameter="n_folds"
    )
    splits[folds[0]] = TEST
    splits[folds[1]] = VAL

    train_pool = development.drop(index=folds[0].union(folds[1]))
    calibration = _grouped_folds(
        train_pool,
        n_splits=config.calibration_folds,
        seed=config.seed,
        parameter="calibration_folds",
    )[0]
    splits[calibration] = CALIB
    splits[train_pool.drop(index=calibration).index] = TRAIN

    unassigned = int(splits.isna().sum())
    if unassigned:
        raise SplitError(f"{unassigned} rows were not assigned a split")
    return splits


def grouping_key(frame: pd.DataFrame) -> pd.Series:
    """The unit that must not straddle a split.

    Patient and duplicate cluster are two overlapping relations, not one:
    a patient can own an image that also has a twin in another collection under
    a different patient id. Taking either key alone leaves the other's leakage
    in place, and taking them as separate columns is not something a grouped
    splitter can express. So they are merged into connected components — if two
    rows share a patient *or* a cluster, they land in the same split, and so
    does anything transitively reachable from them.

    Falls back to patient id alone when the manifest has not been annotated,
    which keeps this usable on a corpus that has not been hashed yet.
    """
    if DUPLICATE_GROUP not in frame.columns:
        return frame["patient_id"]

    parent: dict[str, str] = {}

    def find(node: str) -> str:
        parent.setdefault(node, node)
        while parent[node] != node:
            parent[node] = parent[parent[node]]
            node = parent[node]
        return node

    def union(left: str, right: str) -> None:
        left_root, right_root = find(left), find(right)
        if left_root != right_root:
            parent[max(left_root, right_root)] = min(left_root, right_root)

    patients = frame["patient_id"].astype("string")
    clusters = "dup:" + frame[DUPLICATE_GROUP].astype("string")
    for patient, cluster_id in zip(patients, clusters, strict=True):
        union(str(patient), str(cluster_id))

    return pd.Series(
        [find(str(patient)) for patient in patients], index=frame.index, dtype="string"
    )


def _grouped_folds(
    frame: pd.DataFrame, *, n_splits: int, seed: int, parameter: str = "n_folds"
) -> list[pd.Index]:
    """Patient-grouped, label-stratified fold indices.

    Deliberately raises rather than clamping `n_splits` to something feasible:
    silently shrinking the calibration slice would change how well temperature
    scaling can be fitted without anyone being told.
    """
    groups = grouping_key(frame)
    labels = frame["label"]

    smallest_class = int(labels.value_counts().min())
    distinct_patients = int(groups.nunique())
    if n_splits > distinct_patients:
        raise SplitError(
            f"cannot make {n_splits} folds from {distinct_patients} distinct groups; "
            f"lower {parameter}"
        )
    if n_splits > smallest_class:
        raise SplitError(
            f"cannot make {n_splits} folds when the smallest class has {smallest_class} rows; "
            f"lower {parameter} or rebalance the corpus"
        )

    splitter = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    return [
        frame.index[positions]
        for _, positions in splitter.split(frame, y=labels, groups=groups)
    ]


def leave_one_source_out(frame: pd.DataFrame) -> Iterator[tuple[str, pd.Index, pd.Index]]:
    """Yield (held-out source, train index, test index) for every source.

    The spread across these is more informative than any single external
    number: a model that scores well on four sources and collapses on the fifth
    has learned four acquisition pipelines, not a disease.
    """
    for source in sorted(set(frame["source"])):
        held_out = frame["source"] == source
        if not held_out.any():
            continue
        yield source, frame.index[~held_out], frame.index[held_out]


def summarise(frame: pd.DataFrame, splits: pd.Series) -> pd.DataFrame:
    """Rows per split per class, with patient counts — read this before training."""
    working = frame.assign(split=splits)
    table = (
        working.pivot_table(index="split", columns="label", values="image_id", aggfunc="count")
        .fillna(0)
        .astype(int)
    )
    table["images"] = table.sum(axis=1)
    table["patients"] = working.groupby("split")["patient_id"].nunique()
    table["sources"] = working.groupby("split")["source"].nunique()
    return table.reindex([name for name in SPLIT_NAMES if name in table.index])
