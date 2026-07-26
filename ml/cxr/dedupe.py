"""Near-duplicate grouping, and the choice of which copy to keep.

G2 exists because the public collections are aggregates of each other. On the
first real corpus it found exactly that, at a scale worth recording: 8,850 of
the RSNA Pneumonia Challenge's 8,851 normal studies have a pixel-identical twin
in the COVID-19 Radiography Database's normal class. Every one of those 8,850
clusters holds exactly two images, both labelled normal, with no label
conflicts anywhere in the set.

That is not the hash over-clustering. The two copies travel completely
different decode paths -- a 299x299 PNG re-encode on one side, a 1024x1024
DICOM through the VOI LUT on the other -- and still land on the same 256-bit
signature. Agreement that survives both paths is agreement about the image.

Left alone this leaks: the twins land in different splits and the test set
measures memorisation. Grouping them is handled in `splits`; this module is
about identifying the groups and, when asked, dropping the redundant copies.
"""

from __future__ import annotations

import pandas as pd

from cxr.hashing import DEFAULT_THRESHOLD_BITS, cluster

GROUP_COLUMN = "duplicate_group"

# Which copy survives deduplication, best first. RSNA ships the DICOM original
# at full resolution with the acquisition header intact; the COVID-19
# Radiography Database ships a 299x299 re-encode of it with the header gone.
#
# It also happens to be the choice that makes the corpus less confounded rather
# than more. Dropping the RSNA copies instead would leave the normal class
# almost entirely inside covid_radiography and the pneumonia class almost
# entirely inside rsna_pneumonia, which hands the model source as a near
# -perfect proxy for label. Preferring the DICOM keeps both non-COVID classes
# genuinely mixed across sources. It does nothing for the COVID class, which is
# single-sourced by construction -- no pre-pandemic dataset can fix that.
DEFAULT_PREFERENCE: tuple[str, ...] = ("rsna_pneumonia", "covid_radiography")


def annotate(frame: pd.DataFrame, *, threshold: int = DEFAULT_THRESHOLD_BITS) -> pd.DataFrame:
    """Return `frame` with a `duplicate_group` column.

    Every row gets a group, including singletons, so downstream code can treat
    the column as a grouping key without special-casing.
    """
    usable = frame[frame["phash"].notna()]
    assignments = cluster(
        zip(usable["image_id"], usable["phash"], strict=True), threshold=threshold
    )
    groups = frame["image_id"].map(assignments)

    # A row with no perceptual hash is its own group rather than a member of a
    # shared NaN bucket, which would silently glue unrelated images together.
    orphans = groups.isna()
    if orphans.any():
        groups = groups.astype("object")
        offset = int(groups[~orphans].max()) + 1 if (~orphans).any() else 0
        groups[orphans] = range(offset, offset + int(orphans.sum()))
    return frame.assign(**{GROUP_COLUMN: groups.astype("int64")})


def summarise(frame: pd.DataFrame) -> pd.DataFrame:
    """Per-group composition of the clusters that hold more than one image."""
    if GROUP_COLUMN not in frame.columns:
        frame = annotate(frame)
    sizes = frame.groupby(GROUP_COLUMN)["image_id"].transform("count")
    multi = frame[sizes > 1]
    if multi.empty:
        return pd.DataFrame(columns=["clusters", "images", "cross_source", "label_conflicts"])

    per_cluster = multi.groupby(GROUP_COLUMN)
    return pd.DataFrame(
        {
            "clusters": [int(per_cluster.ngroups)],
            "images": [len(multi)],
            "cross_source": [int((per_cluster["source"].nunique() > 1).sum())],
            "label_conflicts": [int((per_cluster["label"].nunique() > 1).sum())],
        }
    )


def deduplicate(
    frame: pd.DataFrame,
    *,
    preference: tuple[str, ...] = DEFAULT_PREFERENCE,
    threshold: int = DEFAULT_THRESHOLD_BITS,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Keep one representative per duplicate group. Returns (kept, dropped).

    Groups whose members disagree on the label are kept whole and untouched.
    A disagreement means either the hash is wrong or one of the datasets is,
    and resolving that by silently discarding a row would bury the evidence.
    Splitting still groups them, so they cannot leak either way.
    """
    if GROUP_COLUMN not in frame.columns:
        frame = annotate(frame, threshold=threshold)

    rank = {name: index for index, name in enumerate(preference)}
    ordered = frame.assign(
        _rank=frame["source"].map(lambda name: rank.get(name, len(preference))),
    ).sort_values(["_rank", "image_id"], kind="stable")

    conflicted = ordered.groupby(GROUP_COLUMN)["label"].transform("nunique") > 1
    resolvable = ordered[~conflicted]
    keep_index = resolvable.groupby(GROUP_COLUMN, sort=False).head(1).index

    protected = conflicted.reindex(frame.index, fill_value=False)
    kept = frame.loc[frame.index.isin(keep_index) | protected]
    dropped = frame.loc[~frame.index.isin(kept.index)]
    return kept.copy(), dropped.copy()
