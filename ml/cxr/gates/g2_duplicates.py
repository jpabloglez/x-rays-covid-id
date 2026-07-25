"""G2 — no near-duplicate image cluster spans two splits.

The public COVID collections are aggregates of each other. The same radiograph
turns up in several of them, re-encoded and resized, under unrelated filenames
and with no shared patient id — so G1 cannot see it. Exact hashing cannot see
it either, because the bytes differ.
"""

from __future__ import annotations

import pandas as pd

from cxr.gates.base import GateResult, GateStatus
from cxr.hashing import DEFAULT_THRESHOLD_BITS, cluster

GATE = "G2"
TITLE = "Near-duplicate disjointness"

DEFAULT_THRESHOLD = DEFAULT_THRESHOLD_BITS


def near_duplicate_disjointness(
    frame: pd.DataFrame,
    splits: pd.Series,
    *,
    threshold: int = DEFAULT_THRESHOLD,
    ignore: tuple[str, ...] = (),
) -> GateResult:
    """Fail if a perceptual-hash cluster contains images from two splits."""
    working = pd.DataFrame(
        {
            "image_id": frame["image_id"],
            "sha256": frame["sha256"],
            "phash": frame["phash"],
            "source": frame["source"],
            "split": splits,
        }
    )
    working = working[~working["split"].isin(ignore)]

    usable = working[working["phash"].notna()]
    if usable.empty:
        return GateResult(
            gate=GATE,
            title=TITLE,
            status=GateStatus.SKIPPED,
            summary="No perceptual hashes in the manifest; run the hashing step first.",
        )

    assignments = cluster(
        zip(usable["image_id"], usable["phash"], strict=True), threshold=threshold
    )
    usable = usable.assign(cluster=usable["image_id"].map(assignments))

    per_cluster = usable.groupby("cluster")
    spans = per_cluster["split"].nunique()
    straddling = spans[spans > 1]

    exact_duplicates = int(usable["sha256"].duplicated().sum())
    near_duplicate_images = int((per_cluster["image_id"].transform("count") > 1).sum())

    details = {
        "threshold_bits": threshold,
        "clusters": int(spans.size),
        "images_in_multi_image_clusters": near_duplicate_images,
        "exact_duplicate_images": exact_duplicates,
    }

    if len(straddling):
        offending = usable[usable["cluster"].isin(straddling.index)]
        examples = [
            {
                "images": group["image_id"].tolist()[:4],
                "splits": sorted(set(group["split"])),
                "sources": sorted(set(group["source"])),
            }
            for _, group in list(offending.groupby("cluster"))[:3]
        ]
        return GateResult(
            gate=GATE,
            title=TITLE,
            status=GateStatus.FAIL,
            summary=(
                f"{len(straddling)} near-duplicate clusters span more than one split, "
                f"covering {len(offending)} images. Examples: {examples}"
            ),
            measured=float(len(straddling)),
            threshold=0.0,
            details={**details, "straddling_clusters": int(len(straddling))},
        )

    return GateResult(
        gate=GATE,
        title=TITLE,
        status=GateStatus.PASS,
        summary=(
            f"{int(spans.size)} clusters at {threshold}-bit distance; "
            f"{near_duplicate_images} images sit in a multi-image cluster, none across splits. "
            f"{exact_duplicates} exact duplicates."
        ),
        measured=0.0,
        threshold=0.0,
        details=details,
    )
