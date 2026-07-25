"""Leakage and confound gates.

Each gate is an assertion about an assembled manifest that runs in CI and
fails the build. They exist because every one of them corresponds to a way the
published COVID-classifier literature produced numbers that did not survive
contact with a second hospital.

A gate returns a GateResult rather than a bare boolean: the measured value is
what goes in the model card, and "G4 passed" is far less useful than "class and
source have Cramer's V 0.31, below the declared 0.40 threshold".
"""

from cxr.gates.base import GateResult, GateStatus
from cxr.gates.g1_patient import patient_disjointness
from cxr.gates.g2_duplicates import near_duplicate_disjointness
from cxr.gates.g3_source_probe import source_confound_probe
from cxr.gates.g4_class_source import class_source_independence
from cxr.gates.runner import run_all

__all__ = [
    "GateResult",
    "GateStatus",
    "class_source_independence",
    "near_duplicate_disjointness",
    "patient_disjointness",
    "run_all",
    "source_confound_probe",
]
