"""G5, which measures what G4 cannot see on a single-source corpus."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from cxr.gates.base import GateStatus
from cxr.gates.g5_scanner import (
    attainable_balanced_accuracy,
    cramers_v,
    scanner_independence,
)


def _frame(rows):
    """rows: (manufacturer, scanner_model, label, count)."""
    records = []
    for manufacturer, model, label, count in rows:
        records += [
            {"manufacturer": manufacturer, "scanner_model": model, "label": label}
        ] * count
    return pd.DataFrame(records)


def test_a_device_that_sorts_the_classes_fails():
    """The failure this gate exists for: portable units on the COVID ward,
    the radiology suite for scheduled outpatients. One source throughout, and
    the machine still gives the answer away."""
    frame = _frame([
        ("Portable", "MobileDR", "covid", 200),
        ("Suite", "FixedDR", "non_covid", 200),
    ])
    result = scanner_independence(frame)
    assert result.status is GateStatus.FAIL
    assert result.measured > 0.9
    assert result.details["manufacturer"]["attainable_balanced_accuracy"] == pytest.approx(1.0)


def test_devices_used_evenly_across_classes_pass():
    frame = _frame([
        ("Philips", "DigitalDiagnost", "covid", 100),
        ("Philips", "DigitalDiagnost", "non_covid", 100),
        ("Siemens", "FD-X", "covid", 100),
        ("Siemens", "FD-X", "non_covid", 100),
    ])
    result = scanner_independence(frame)
    assert result.status is GateStatus.PASS
    assert result.measured == pytest.approx(0.0, abs=0.05)


def test_the_attainable_accuracy_is_chance_when_the_device_says_nothing():
    frame = _frame([
        ("A", "a", "covid", 50), ("A", "a", "non_covid", 50),
        ("B", "b", "covid", 50), ("B", "b", "non_covid", 50),
    ])
    assert attainable_balanced_accuracy(frame, "manufacturer") == pytest.approx(0.5)


def test_a_corpus_with_no_device_recorded_skips_rather_than_passing():
    """An absence of data is not evidence of independence, and a gate that
    reported PASS here would be claiming a measurement it never made."""
    frame = pd.DataFrame({
        "manufacturer": pd.Series([None] * 8, dtype="string"),
        "scanner_model": pd.Series([None] * 8, dtype="string"),
        "label": ["covid"] * 4 + ["non_covid"] * 4,
    })
    result = scanner_independence(frame)
    assert result.status is GateStatus.SKIPPED
    assert "not evidence of independence" in result.summary


def test_unrecorded_devices_are_a_level_not_a_deletion():
    """Which images lack a scanner is itself an acquisition signature. Dropping
    those rows would hide a confound rather than measure it."""
    frame = pd.DataFrame({
        "manufacturer": pd.Series(["A"] * 100 + [None] * 100, dtype="string"),
        "scanner_model": pd.Series(["a"] * 100 + [None] * 100, dtype="string"),
        "label": ["covid"] * 100 + ["non_covid"] * 100,
    })
    result = scanner_independence(frame)
    assert result.status is GateStatus.FAIL
    assert result.details["manufacturer"]["unrecorded"] == 100


def test_the_correction_stops_many_thin_levels_inventing_a_confound():
    """Uncorrected Cramer's V grows with the number of levels whether or not any
    association exists. Here every device is balanced across the classes, so a
    corrected estimate has to stay near zero however many devices there are."""
    frame = _frame([(f"dev{i}", f"dev{i}", label, 5)
                    for i in range(40) for label in ("covid", "non_covid")])
    table = pd.crosstab(frame["manufacturer"], frame["label"])
    assert cramers_v(table) < 0.2


def test_one_image_per_device_is_unestimable_and_must_not_read_as_a_pass():
    """With a single observation per device the correction has no degrees of
    freedom and returns NaN. `nan > threshold` is False, so an unguarded gate
    would report PASS on a corpus it could not measure -- the same error as
    counting a skipped gate as a passed one."""
    rng = np.random.default_rng(0)
    frame = pd.DataFrame({
        "manufacturer": pd.Series([f"dev{i}" for i in range(200)], dtype="string"),
        "scanner_model": pd.Series([f"dev{i}" for i in range(200)], dtype="string"),
        "label": rng.choice(["covid", "non_covid"], size=200),
    })
    assert np.isnan(cramers_v(pd.crosstab(frame["manufacturer"], frame["label"])))

    result = scanner_independence(frame)
    assert result.status is GateStatus.SKIPPED
    assert "not measured, so not cleared" in result.summary.lower()


def test_the_worst_grain_is_the_one_reported():
    """Manufacturer can look clean while the individual model does not, and
    reporting the reassuring number would defeat the point."""
    frame = _frame([
        ("Philips", "Portable", "covid", 100),
        ("Philips", "Suite", "non_covid", 100),
        ("Siemens", "Portable2", "covid", 100),
        ("Siemens", "Suite2", "non_covid", 100),
    ])
    result = scanner_independence(frame)
    # Manufacturer is balanced across classes; the model is perfectly sorted.
    assert result.details["manufacturer"]["cramers_v"] == pytest.approx(0.0, abs=0.05)
    assert result.details["scanner_model"]["cramers_v"] > 0.9
    assert result.status is GateStatus.FAIL
    assert "scanner_model" in result.summary
