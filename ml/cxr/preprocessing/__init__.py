"""Preprocessing, declared as data and executed two ways.

The spec is the contract; `reference` defines what it means in numpy and PIL
alone, and `monai_pipeline` executes the same spec through MONAI for training.
An equivalence test keeps them honest, so train/serve drift fails the build
instead of quietly degrading the deployed model.
"""

from cxr.preprocessing.reference import apply, load_grayscale
from cxr.preprocessing.spec import SPEC_VERSION, PreprocessingSpec, ResizeMode, Windowing

__all__ = [
    "SPEC_VERSION",
    "PreprocessingSpec",
    "ResizeMode",
    "Windowing",
    "apply",
    "load_grayscale",
]
