"""What every dataset adapter has to provide.

An adapter's job is narrow: turn a downloaded directory into manifest records
and refuse to guess. Where a dataset does not record something — and the
important one is patient identity — the adapter says so rather than inventing
a value, because a fabricated patient id turns G1 into a gate that always
passes.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path

from cxr.hashing import dhash, sha256_file


class SourceError(RuntimeError):
    """Raised when a dataset directory is not laid out as the adapter expects."""


@dataclass(frozen=True)
class SourceInfo:
    """What a source is and what it can honestly support."""

    name: str
    title: str
    citation: str
    licence: str
    has_covid_label: bool
    has_patient_ids: bool
    has_demographics: bool
    modality: str
    notes: str = ""


class Source(ABC):
    """Adapter turning one downloaded dataset into manifest records."""

    info: SourceInfo

    @abstractmethod
    def build(self, root: Path) -> list[dict]:
        """Scan `root` and return manifest records with paths relative to it."""

    def hash_record(self, absolute: Path, relative: Path) -> dict:
        """The fields every adapter fills the same way."""
        from PIL import Image

        with Image.open(absolute) as image:
            width, height = image.size
            phash = dhash(image)
        return {
            "path": str(relative),
            "sha256": sha256_file(absolute),
            "phash": phash,
            "width": width,
            "height": height,
        }

    def synthetic_patient_id(self, index: int) -> str:
        """One image, one patient — used only where the dataset has no ids.

        This is not a fix. It means G1 cannot detect same-patient leakage for
        this source, because the information required to detect it was never
        published. Record it in the model card as a known blind spot.
        """
        if self.info.has_patient_ids:
            raise SourceError(
                f"{self.info.name} publishes patient ids; use them rather than synthesising"
            )
        return f"{self.info.name}:img{index:07d}"
