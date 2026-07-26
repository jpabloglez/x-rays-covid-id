"""Acknowledging a confound is not the same as ignoring a defect."""

from __future__ import annotations

from cxr.gates.base import GateResult, GateStatus
from cxr.gates.runner import BLOCKING_GATES, evaluate, render, to_json


def _result(gate, status, measured=None):
    return GateResult(
        gate=gate,
        title=gate,
        status=status,
        summary=f"{gate} {status}",
        measured=measured,
        threshold=0.5,
    )


PASSING = [_result(name, GateStatus.PASS) for name in ("G1", "G1b", "G2")]
CONFOUNDED = [
    *PASSING,
    _result("G3", GateStatus.FAIL, 0.795),
    _result("G4", GateStatus.FAIL, 0.699),
]
LEAKING = [
    _result("G1", GateStatus.PASS),
    _result("G1b", GateStatus.PASS),
    _result("G2", GateStatus.FAIL, 5676),
    _result("G3", GateStatus.FAIL, 0.795),
]


def test_clean_gates_permit_training():
    assert evaluate(PASSING).ok


def test_an_unacknowledged_confound_blocks():
    verdict = evaluate(CONFOUNDED)
    assert not verdict.ok
    assert [result.gate for result in verdict.blocking] == ["G3", "G4"]


def test_acknowledging_the_confounds_permits_training():
    """Track 1 trains on a knowingly confounded corpus on purpose."""
    verdict = evaluate(CONFOUNDED, {"G3", "G4"})
    assert verdict.ok
    assert [result.gate for result in verdict.acknowledged] == ["G3", "G4"]


def test_acknowledging_only_one_still_blocks_on_the_other():
    verdict = evaluate(CONFOUNDED, {"G3"})
    assert not verdict.ok
    assert [result.gate for result in verdict.blocking] == ["G4"]


def test_a_leakage_gate_cannot_be_acknowledged():
    """An image on both sides of a split invalidates everything measured after
    it, so there is no experiment this could be waived for."""
    verdict = evaluate(LEAKING, {"G2", "G3"})
    assert not verdict.ok
    assert [result.gate for result in verdict.blocking] == ["G2"]
    assert "G2" in BLOCKING_GATES


def test_a_stale_acknowledgement_is_itself_an_error():
    """Otherwise a fixed corpus keeps a live check silently disarmed."""
    verdict = evaluate(CONFOUNDED, {"G3", "G4", "G1"})
    assert not verdict.ok
    assert verdict.stale == ["G1"]


def test_an_unknown_gate_name_is_an_error():
    verdict = evaluate(CONFOUNDED, {"G9"})
    assert verdict.unknown == ["G9"]
    assert not verdict.ok


def test_render_explains_how_to_acknowledge_a_confound():
    text = render(CONFOUNDED)
    assert "--acknowledge" in text
    assert "G3" in text and "G4" in text


def test_render_does_not_offer_to_acknowledge_a_leakage_gate():
    offer = [line for line in render(LEAKING).splitlines() if "--acknowledge" in line]
    assert offer and "G2" not in offer[0]


def test_render_names_the_stale_entry():
    assert "G1 is acknowledged" in render(CONFOUNDED, {"G3", "G4", "G1"})


def test_json_records_what_was_acknowledged(tmp_path):
    """A model card reporting a confounded corpus has to say the confound was
    known in advance, or it is describing different work."""
    import json

    path = tmp_path / "gates.json"
    to_json(CONFOUNDED, path, {"G3", "G4"})
    payload = json.loads(path.read_text())

    assert payload["acknowledged"] == ["G3", "G4"]
    assert payload["blocking"] == []
    assert payload["training_permitted"] is True
    assert {row["gate"] for row in payload["gates"] if row["acknowledged"]} == {"G3", "G4"}


def test_json_marks_a_blocked_run_as_not_permitted(tmp_path):
    import json

    path = tmp_path / "gates.json"
    to_json(LEAKING, path, {"G3"})
    payload = json.loads(path.read_text())

    assert payload["blocking"] == ["G2"]
    assert payload["training_permitted"] is False
