"""G4 — how much of the label is predictable from provenance alone?

G3 asks whether a model can tell the sources apart. G4 asks whether that helps
it. Together they bound the shortcut: if source is trivially visible *and*
source predicts class, then a network can score well without ever looking at a
lung, and it will, because that path is cheaper than learning radiology.

Measured with Cramer's V over the class-by-source contingency table. V is 0
when class and source are independent and 1 when either determines the other.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from cxr.gates.base import GateResult, GateStatus
from cxr.splits import EXTERNAL

GATE = "G4"
TITLE = "Class-source independence"

DEFAULT_THRESHOLD = 0.40

# Below this expected cell count the chi-square approximation stops holding.
MIN_EXPECTED_COUNT = 5.0


def _smallest_expected_count(table: np.ndarray) -> float:
    total = table.sum()
    if total == 0:
        return 0.0
    expected = table.sum(axis=1, keepdims=True) @ table.sum(axis=0, keepdims=True) / total
    return float(expected.min())


def cramers_v(table: np.ndarray) -> float:
    """Cramer's V for a contingency table, bias-corrected (Bergsma, 2013).

    The correction matters here: with few sources and few classes the
    uncorrected statistic is optimistic, and this number goes in a model card.
    """
    total = table.sum()
    if total == 0:
        return 0.0
    row_totals = table.sum(axis=1, keepdims=True)
    column_totals = table.sum(axis=0, keepdims=True)
    expected = row_totals @ column_totals / total
    with np.errstate(divide="ignore", invalid="ignore"):
        terms = np.where(expected > 0, (table - expected) ** 2 / expected, 0.0)
    chi2 = float(terms.sum())

    rows, columns = table.shape
    if min(rows, columns) < 2:
        return 0.0

    phi2 = chi2 / total
    phi2_corrected = max(0.0, phi2 - (rows - 1) * (columns - 1) / (total - 1))
    rows_corrected = rows - (rows - 1) ** 2 / (total - 1)
    columns_corrected = columns - (columns - 1) ** 2 / (total - 1)
    denominator = min(rows_corrected - 1, columns_corrected - 1)
    if denominator <= 0:
        return 0.0
    return float(np.sqrt(phi2_corrected / denominator))


def class_source_independence(
    frame: pd.DataFrame,
    splits: pd.Series | None = None,
    *,
    threshold: float = DEFAULT_THRESHOLD,
) -> GateResult:
    """Fail if class and source are associated above `threshold` in any split.

    Evaluated per split rather than once over the pool, because a corpus can
    look balanced overall while a single split is entirely confounded.
    """
    if frame["source"].nunique() < 2:
        return GateResult(
            gate=GATE,
            title=TITLE,
            status=GateStatus.SKIPPED,
            summary="Only one source in the manifest; class and source cannot be associated.",
        )

    overall = cramers_v(pd.crosstab(frame["label"], frame["source"]).to_numpy(dtype=float))

    per_split: dict[str, float] = {}
    underpowered: list[str] = []
    if splits is not None:
        for name in sorted(set(splits.dropna())):
            if name == EXTERNAL:
                # The external split is one source by construction; V is
                # undefined there and its absence is not a finding.
                continue
            subset = frame[splits == name]
            if subset["source"].nunique() < 2 or subset["label"].nunique() < 2:
                continue
            table = pd.crosstab(subset["label"], subset["source"]).to_numpy(dtype=float)
            if _smallest_expected_count(table) < MIN_EXPECTED_COUNT:
                # Chi-square, and therefore V, is unreliable once an expected
                # cell drops below about five. A small calibration slice will
                # otherwise report a large association that is pure sampling
                # noise, and a gate that cries wolf gets switched off.
                underpowered.append(str(name))
                continue
            per_split[str(name)] = cramers_v(table)

    single_source_classes = [
        str(label)
        for label, count in frame.groupby("label")["source"].nunique().items()
        if count == 1
    ]

    worst_name, worst_value = "overall", overall
    for name, value in per_split.items():
        if value > worst_value:
            worst_name, worst_value = name, value

    details = {
        "overall": round(overall, 4),
        "per_split": {name: round(value, 4) for name, value in per_split.items()},
        "splits_too_small_to_score": underpowered,
        "classes_from_a_single_source": single_source_classes,
        "sources": sorted(set(frame["source"].astype(str))),
    }

    if single_source_classes:
        return GateResult(
            gate=GATE,
            title=TITLE,
            status=GateStatus.FAIL,
            summary=(
                f"Classes {single_source_classes} are drawn from exactly one source each. "
                f"Recognising the source is sufficient to predict those classes, whatever "
                f"Cramer's V says (measured {worst_value:.3f} at {worst_name})."
            ),
            measured=worst_value,
            threshold=threshold,
            details=details,
        )

    if worst_value > threshold:
        return GateResult(
            gate=GATE,
            title=TITLE,
            status=GateStatus.FAIL,
            summary=(
                f"Class and source are associated at Cramer's V {worst_value:.3f} "
                f"({worst_name}), above the declared {threshold:.2f}. That fraction of the "
                f"label is available from provenance alone."
            ),
            measured=worst_value,
            threshold=threshold,
            details=details,
        )

    return GateResult(
        gate=GATE,
        title=TITLE,
        status=GateStatus.PASS,
        summary=(
            f"Worst class-source association is Cramer's V {worst_value:.3f} at "
            f"{worst_name}, below the declared {threshold:.2f}."
        ),
        measured=worst_value,
        threshold=threshold,
        details=details,
    )
