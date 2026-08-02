"""G5: within a single collection, is the class predictable from the scanner?

G4 asks whether class is confounded with the *collection*. On Track 2 it cannot
answer -- there is one collection, so the association is undefined and the gate
skips. That skip is honest but it is not reassurance, and reading it as one is
the mistake this module exists to prevent: a corpus can be perfectly clean at
the collection level and still hand a model the answer through the acquisition
device.

The mechanism is ordinary hospital logistics rather than anything exotic. A
COVID ward gets a portable unit wheeled to the bedside; a scheduled outpatient
chest film goes through the radiology suite. Those are different machines with
different detectors, different processing pipelines and different noise, and if
the wards were sorted by infection status then the device is a label proxy that
no amount of single-source purity removes.

Two numbers, because they answer different questions:

- **Association.** Cramér's V between scanner and class. High means the device
  correlates with the label in the data, whatever a model does with it.
- **Attainable accuracy.** Balanced accuracy of the best possible constant
  guess per device -- what a model would score by reading the machine and
  nothing else. This is the number that says how much of the headline score
  could come for free.

Reported at two grains. Manufacturer is coarse and has enough rows per level
to be stable; the individual model is finer and closer to what a network could
actually key on, at the cost of thin cells.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from cxr.gates.base import GateResult, GateStatus

GATE = "G5"
TITLE = "Scanner-class independence"
DEFAULT_THRESHOLD = 0.40

UNRECORDED = "unrecorded"


def _labels(frame: pd.DataFrame, column: str) -> pd.Series:
    """Device identity, with missing values kept as their own level.

    Dropping unrecorded rows would be the wrong move twice over: it shrinks the
    sample, and "which images have no scanner recorded" is itself an
    acquisition signature that could correlate with the class.
    """
    return frame[column].fillna(UNRECORDED).astype(str)


def cramers_v(table: pd.DataFrame) -> float:
    """Bias-corrected Cramér's V.

    The correction matters here. With 65 device strings over 1,463 rows many
    cells are nearly empty, and uncorrected V rises with the number of levels
    whether or not any association exists -- it would report a confound created
    by counting.
    """
    from scipy.stats import chi2_contingency

    counts = table.to_numpy()
    total = counts.sum()
    if total == 0 or min(counts.shape) < 2:
        return float("nan")
    chi2 = chi2_contingency(counts)[0]
    phi2 = chi2 / total
    rows, columns = counts.shape
    phi2_corrected = max(0.0, phi2 - (rows - 1) * (columns - 1) / (total - 1))
    rows_corrected = rows - (rows - 1) ** 2 / (total - 1)
    columns_corrected = columns - (columns - 1) ** 2 / (total - 1)
    denominator = min(rows_corrected - 1, columns_corrected - 1)
    return float(np.sqrt(phi2_corrected / denominator)) if denominator > 0 else float("nan")


def attainable_balanced_accuracy(frame: pd.DataFrame, column: str) -> float:
    """Balanced accuracy from the device alone, guessing the majority per device.

    An upper bound on what "read the machine, ignore the chest" achieves. It is
    optimistic on purpose -- it assumes the mapping is learned perfectly -- so
    a low value is genuine reassurance while a high one is only a ceiling.
    """
    devices = _labels(frame, column)
    truth = frame["label"].astype(str)
    classes = sorted(truth.unique())
    if len(classes) < 2:
        return float("nan")

    predicted = pd.Series(index=frame.index, dtype=object)
    for _device, rows in truth.groupby(devices):
        predicted.loc[rows.index] = rows.value_counts().idxmax()

    recalls = [
        float((predicted[truth == name] == name).mean())
        for name in classes
        if int((truth == name).sum()) > 0
    ]
    return float(np.mean(recalls)) if recalls else float("nan")


def summarise(frame: pd.DataFrame, column: str) -> dict:
    devices = _labels(frame, column)
    table = pd.crosstab(devices, frame["label"].astype(str))
    return {
        "levels": int(devices.nunique()),
        "unrecorded": int((devices == UNRECORDED).sum()),
        "cramers_v": cramers_v(table),
        "attainable_balanced_accuracy": attainable_balanced_accuracy(frame, column),
    }


def scanner_independence(
    frame: pd.DataFrame, *, threshold: float = DEFAULT_THRESHOLD
) -> GateResult:
    """Run G5, or skip when the corpus records no device at all."""
    columns = [name for name in ("manufacturer", "scanner_model") if name in frame.columns]
    recorded = [name for name in columns if frame[name].notna().any()]
    if not recorded:
        return GateResult(
            gate=GATE,
            title=TITLE,
            status=GateStatus.SKIPPED,
            summary=(
                "No acquisition device recorded, so scanner and class cannot be "
                "associated. This is an absence of data, not evidence of independence."
            ),
            measured=None,
            threshold=threshold,
            details={"columns_present": columns},
        )

    details = {name: summarise(frame, name) for name in recorded}

    # NaN is a real outcome here, not a bug to paper over: with one image per
    # device the correction has no degrees of freedom left and the association
    # is genuinely unestimable. It must not reach the comparison below, because
    # `nan > threshold` is False and the gate would report PASS on a corpus it
    # could not measure at all -- the same mistake as counting a skipped gate
    # as a passed one.
    estimable = {
        name: entry for name, entry in details.items() if not np.isnan(entry["cramers_v"])
    }
    if not estimable:
        return GateResult(
            gate=GATE,
            title=TITLE,
            status=GateStatus.SKIPPED,
            summary=(
                "Scanner-class association is unestimable: too few images per device for "
                "any grain to support a corrected estimate. Not measured, so not cleared."
            ),
            measured=None,
            threshold=threshold,
            details=details,
        )

    grain = max(estimable, key=lambda name: estimable[name]["cramers_v"])
    measured = estimable[grain]["cramers_v"]
    failed = measured > threshold
    return GateResult(
        gate=GATE,
        title=TITLE,
        status=GateStatus.FAIL if failed else GateStatus.PASS,
        summary=(
            f"Cramer's V {measured:.3f} between {grain} and class over "
            f"{details[grain]['levels']} devices; guessing from the device alone reaches "
            f"{details[grain]['attainable_balanced_accuracy']:.1%} balanced accuracy."
        ),
        measured=float(measured),
        threshold=threshold,
        details=details,
    )
