"""Run every gate and report as a set.

Gates run to completion rather than short-circuiting on the first failure: the
useful output is the whole picture, and G3 and G4 in particular are read
together — either alone is much weaker evidence than the pair.

Two kinds of failure are deliberately not treated the same. G1, G1b and G2 are
defects: an image on both sides of a split makes every number downstream
meaningless, and no experiment needs that. G3 and G4 are findings — a corpus
assembled from a pre-pandemic pneumonia set and a pandemic-era COVID set is
confounded by construction, and Track 1 exists precisely to train on it and
measure what the confound is worth. Blocking that would be blocking the
experiment.

So confound gates can be acknowledged by name, and acknowledgement is checked
rather than trusted: naming a gate that then passes is itself an error, because
a stale acknowledgement silently disarms a live check.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from cxr.gates.base import GateResult, GateStatus
from cxr.gates.g1_patient import external_is_source_pure, patient_disjointness
from cxr.gates.g2_duplicates import near_duplicate_disjointness
from cxr.gates.g3_source_probe import source_confound_probe
from cxr.gates.g4_class_source import class_source_independence
from cxr.hashing import DEFAULT_THRESHOLD_BITS


def run_all(
    frame: pd.DataFrame,
    splits: pd.Series,
    *,
    image_root: Path | dict[str, Path] | None = None,
    features: np.ndarray | None = None,
    duplicate_threshold: int = DEFAULT_THRESHOLD_BITS,
    source_probe_threshold: float = 0.75,
    cramers_v_threshold: float = 0.40,
) -> list[GateResult]:
    """Run all gates. G3 is skipped when neither pixels nor features are given."""
    results = [
        patient_disjointness(frame, splits),
        external_is_source_pure(frame, splits),
        near_duplicate_disjointness(frame, splits, threshold=duplicate_threshold),
    ]

    if image_root is None and features is None:
        results.append(
            GateResult(
                gate="G3",
                title="Source-confound probe",
                status=GateStatus.SKIPPED,
                summary="No image root or precomputed features supplied; probe not run.",
            )
        )
    else:
        results.append(
            source_confound_probe(
                frame,
                image_root=image_root,
                features=features,
                threshold=source_probe_threshold,
            )
        )

    results.append(
        class_source_independence(frame, splits, threshold=cramers_v_threshold)
    )
    return results


# Leakage defects. An image on both sides of a split invalidates every number
# measured afterwards, so there is no experiment these can be acknowledged for.
BLOCKING_GATES = frozenset({"G1", "G1b", "G2"})


@dataclass(frozen=True)
class Verdict:
    """What the gate set means once acknowledgements are applied."""

    blocking: list[GateResult] = field(default_factory=list)
    acknowledged: list[GateResult] = field(default_factory=list)
    stale: list[str] = field(default_factory=list)
    unknown: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not (self.blocking or self.stale or self.unknown)


def evaluate(results: list[GateResult], acknowledged: set[str] | None = None) -> Verdict:
    """Split failures into blocking defects and acknowledged confounds."""
    named = set(acknowledged or ())
    by_gate = {result.gate: result for result in results}

    unknown = sorted(named - set(by_gate))
    # A gate named as an acknowledged confound that did not fail: either it was
    # fixed and the acknowledgement outlived it, or it never applied. Both mean
    # the list is describing a corpus that no longer exists.
    stale = sorted(name for name in named - set(unknown) if not by_gate[name].failed)

    blocking, waived = [], []
    for result in results:
        if not result.failed:
            continue
        if result.gate in named and result.gate not in BLOCKING_GATES:
            waived.append(result)
        else:
            blocking.append(result)

    return Verdict(blocking=blocking, acknowledged=waived, stale=stale, unknown=unknown)


def render(results: list[GateResult], acknowledged: set[str] | None = None) -> str:
    lines = [result.render() for result in results]
    verdict = evaluate(results, acknowledged)
    failures = [result for result in results if result.failed]
    lines.append("")

    if verdict.unknown:
        lines.append(f"Acknowledged gates that do not exist: {', '.join(verdict.unknown)}.")
    for name in verdict.stale:
        lines.append(
            f"{name} is acknowledged as a known confound but did not fail. "
            "Remove it from the acknowledgement list rather than carrying a check "
            "that is no longer disarming anything."
        )
    if verdict.blocking:
        blocked = ", ".join(result.gate for result in verdict.blocking)
        waivable = [
            result.gate
            for result in verdict.blocking
            if result.gate not in BLOCKING_GATES and result.gate not in (acknowledged or ())
        ]
        lines.append(
            f"{len(verdict.blocking)} of {len(results)} gates block training: {blocked}. "
            "Do not train on this split."
        )
        if waivable:
            lines.append(
                f"Confound gates can be acknowledged with --acknowledge {','.join(waivable)} "
                "if training on a knowingly confounded corpus is the point."
            )
    elif verdict.acknowledged:
        waived = ", ".join(result.gate for result in verdict.acknowledged)
        lines.append(
            f"{len(failures)} of {len(results)} gates failed; {waived} acknowledged as known "
            "confounds. Training may proceed and the measured values belong in the model card."
        )
    else:
        skipped = [result.gate for result in results if result.status is GateStatus.SKIPPED]
        if skipped:
            # A gate that did not run has not cleared anything, and counting it
            # as a pass is how a corpus acquires a clean bill of health nobody
            # measured. Track 2 skips G4 because it has one source: its freedom
            # from source confounding is true by construction, not by test, and
            # the model card has to say which.
            lines.append(
                f"{len(results) - len(skipped)} of {len(results)} gates passed. "
                f"{len(skipped)} did not run: {', '.join(skipped)}. A skipped gate is not "
                "a passed gate -- it measured nothing, and nothing about it can be quoted."
            )
        else:
            lines.append(f"All {len(results)} gates passed.")
    return "\n".join(lines)


def to_json(
    results: list[GateResult], path: Path, acknowledged: set[str] | None = None
) -> None:
    """Write the measured values for the model card to pick up.

    Acknowledgements are recorded alongside the measurements, because a model
    card that reports a confounded corpus without saying the confound was known
    in advance is describing a different piece of work.
    """
    verdict = evaluate(results, acknowledged)
    payload = {
        "gates": [
            {
                "gate": result.gate,
                "title": result.title,
                "status": str(result.status),
                "summary": result.summary,
                "measured": result.measured,
                "threshold": result.threshold,
                "acknowledged": result.gate in (acknowledged or ()),
                "details": result.details,
            }
            for result in results
        ],
        "acknowledged": sorted(acknowledged or ()),
        "blocking": [result.gate for result in verdict.blocking],
        # Listed explicitly so the model card can distinguish a corpus that
        # cleared a check from one where the check could not be run at all.
        # Reading `status` per gate would give the same answer, but a reader
        # counting passes is exactly the reader who will not do that.
        "skipped": [
            result.gate for result in results if result.status is GateStatus.SKIPPED
        ],
        "training_permitted": verdict.ok,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")


def any_failed(results: list[GateResult]) -> bool:
    return any(result.failed for result in results)
