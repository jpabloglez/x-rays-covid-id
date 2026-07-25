"""Run every gate and report as a set.

Gates run to completion rather than short-circuiting on the first failure: the
useful output is the whole picture, and G3 and G4 in particular are read
together — either alone is much weaker evidence than the pair.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from cxr.gates.base import GateResult, GateStatus
from cxr.gates.g1_patient import external_is_source_pure, patient_disjointness
from cxr.gates.g2_duplicates import near_duplicate_disjointness
from cxr.gates.g3_source_probe import source_confound_probe
from cxr.gates.g4_class_source import class_source_independence


def run_all(
    frame: pd.DataFrame,
    splits: pd.Series,
    *,
    image_root: Path | None = None,
    features: np.ndarray | None = None,
    duplicate_threshold: int = 6,
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


def render(results: list[GateResult]) -> str:
    lines = [result.render() for result in results]
    failures = [result for result in results if result.failed]
    lines.append("")
    if failures:
        lines.append(f"{len(failures)} of {len(results)} gates failed. Do not train on this split.")
    else:
        lines.append(f"All {len(results)} gates passed.")
    return "\n".join(lines)


def to_json(results: list[GateResult], path: Path) -> None:
    """Write the measured values for the model card to pick up."""
    payload = [
        {
            "gate": result.gate,
            "title": result.title,
            "status": str(result.status),
            "summary": result.summary,
            "measured": result.measured,
            "threshold": result.threshold,
            "details": result.details,
        }
        for result in results
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")


def any_failed(results: list[GateResult]) -> bool:
    return any(result.failed for result in results)
