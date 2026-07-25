"""G1 — no patient appears in more than one split.

The cheapest gate and the one most often missing. A patient with a follow-up
film on both sides of a split turns the test set into a memorisation check, and
because the two images genuinely differ, nothing about the number looks wrong.
"""

from __future__ import annotations

import pandas as pd

from cxr.gates.base import GateResult, GateStatus
from cxr.splits import EXTERNAL

GATE = "G1"
TITLE = "Patient disjointness"


def patient_disjointness(
    frame: pd.DataFrame, splits: pd.Series, *, ignore: tuple[str, ...] = ()
) -> GateResult:
    """Fail if any patient_id occurs in two or more splits."""
    working = pd.DataFrame({"patient_id": frame["patient_id"], "split": splits})
    working = working[~working["split"].isin(ignore)]

    per_patient = working.groupby("patient_id")["split"].nunique()
    straddling = per_patient[per_patient > 1]

    total_patients = int(per_patient.size)
    offenders = int(straddling.size)

    if offenders:
        examples = straddling.head(5).index.tolist()
        affected = working[working["patient_id"].isin(straddling.index)]
        return GateResult(
            gate=GATE,
            title=TITLE,
            status=GateStatus.FAIL,
            summary=(
                f"{offenders} of {total_patients} patients appear in more than one split, "
                f"covering {len(affected)} images. Examples: {examples}"
            ),
            measured=float(offenders),
            threshold=0.0,
            details={
                "straddling_patients": straddling.index.tolist(),
                "affected_images": int(len(affected)),
            },
        )

    return GateResult(
        gate=GATE,
        title=TITLE,
        status=GateStatus.PASS,
        summary=f"{total_patients} patients, each confined to a single split.",
        measured=0.0,
        threshold=0.0,
        details={"patients": total_patients},
    )


def external_is_source_pure(frame: pd.DataFrame, splits: pd.Series) -> GateResult:
    """Companion check: the external split must be exactly one unseen source.

    An external set that shares a source with training is an internal set with
    a misleading name, and it is an easy mistake to make when adding data.
    """
    external = frame[splits == EXTERNAL]
    if external.empty:
        return GateResult(
            gate=f"{GATE}b",
            title="External split purity",
            status=GateStatus.SKIPPED,
            summary="No external split configured; there is no held-out-source number to report.",
        )

    external_sources = set(external["source"])
    internal_sources = set(frame[splits != EXTERNAL]["source"])
    overlap = external_sources & internal_sources

    if overlap or len(external_sources) != 1:
        return GateResult(
            gate=f"{GATE}b",
            title="External split purity",
            status=GateStatus.FAIL,
            summary=(
                f"External split holds sources {sorted(external_sources)}"
                + (f" which also appear in training: {sorted(overlap)}" if overlap else "")
            ),
            details={"external": sorted(external_sources), "overlap": sorted(overlap)},
        )

    return GateResult(
        gate=f"{GATE}b",
        title="External split purity",
        status=GateStatus.PASS,
        summary=f"External split is {next(iter(external_sources))!r} and appears nowhere else.",
        details={"external": sorted(external_sources)},
    )
