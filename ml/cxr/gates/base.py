"""Shared result type for the gates."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any


class GateStatus(StrEnum):
    PASS = "pass"
    FAIL = "fail"
    SKIPPED = "skipped"


@dataclass(frozen=True)
class GateResult:
    """Outcome of one gate.

    `measured` and `threshold` are as important as `status`: they are what the
    model card publishes, and a gate that passes at 0.39 against a 0.40
    threshold is a different situation from one that passes at 0.05.
    """

    gate: str
    title: str
    status: GateStatus
    summary: str
    measured: float | None = None
    threshold: float | None = None
    details: dict[str, Any] = field(default_factory=dict)

    @property
    def failed(self) -> bool:
        return self.status is GateStatus.FAIL

    def render(self) -> str:
        marker = {GateStatus.PASS: "PASS", GateStatus.FAIL: "FAIL", GateStatus.SKIPPED: "SKIP"}[
            self.status
        ]
        measurement = ""
        if self.measured is not None:
            measurement = f"  [{self.measured:.4g}"
            if self.threshold is not None:
                measurement += f" vs {self.threshold:.4g}"
            measurement += "]"
        return f"{marker}  {self.gate} · {self.title}{measurement}\n      {self.summary}"
