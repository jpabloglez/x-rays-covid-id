"""NIH ChestX-ray14.

Supplies pneumonia and normal, plus the demographics and view position the
bias analysis needs. It has no COVID label and never will — it was released in
2017. Labels are NLP-mined from radiology reports rather than read for this
purpose, which caps the achievable ceiling on the pneumonia class.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from cxr.manifest import Label, LabelProvenance, Sex, View
from cxr.sources.base import Source, SourceError, SourceInfo

METADATA_FILE = "Data_Entry_2017.csv"
NO_FINDING = "No Finding"
PNEUMONIA_FINDING = "Pneumonia"


class ChestXray14(Source):
    info = SourceInfo(
        name="chestxray14",
        title="NIH ChestX-ray14",
        citation="Wang et al. 2017",
        licence="Public domain (NIH Clinical Center)",
        has_covid_label=False,
        has_patient_ids=True,
        has_demographics=True,
        modality="PNG",
        notes=(
            "Labels are NLP-mined from free-text reports, not verified for this "
            "task; the pneumonia class in particular is noisy. Contains "
            "multiple follow-up films per patient, so grouped splitting is "
            "essential rather than optional."
        ),
    )

    def build(self, root: Path) -> list[dict]:
        metadata_path = root / METADATA_FILE
        if not metadata_path.exists():
            raise SourceError(f"{METADATA_FILE} not found under {root}")

        columns = {
            "Image Index": "filename",
            "Finding Labels": "findings",
            "Patient ID": "patient",
            "Patient Age": "age",
            "Patient Gender": "sex",
            "View Position": "view",
        }
        metadata = pd.read_csv(metadata_path)
        missing = set(columns) - set(metadata.columns)
        if missing:
            raise SourceError(f"{METADATA_FILE} is missing columns: {sorted(missing)}")
        metadata = metadata[list(columns)].rename(columns=columns)

        by_filename = {path.name: path for path in root.rglob("*.png")}

        records: list[dict] = []
        for row in metadata.itertuples(index=False):
            label = _map_findings(set(str(row.findings).split("|")))
            if label is None:
                # Some other pathology with no pneumonia. Emphatically not
                # "normal" — relabelling these as normal is a common shortcut
                # that teaches the model that abnormal chests are healthy.
                continue

            image_path = by_filename.get(str(row.filename))
            if image_path is None:
                continue

            relative = image_path.relative_to(root)
            records.append(
                {
                    "image_id": f"{self.info.name}:{row.filename}",
                    "source": self.info.name,
                    "label": str(label),
                    "label_raw": str(row.findings),
                    "label_provenance": str(LabelProvenance.NLP_MINED),
                    "patient_id": f"{self.info.name}:{row.patient}",
                    "study_id": None,
                    "view": _map_view(row.view),
                    "age": _map_age(row.age),
                    "sex": _map_sex(row.sex),
                    "mask_path": None,
                    **self.hash_record(image_path, relative),
                }
            )

        if not records:
            raise SourceError(f"no usable images matched {METADATA_FILE} under {root}")
        return records


def _map_findings(findings: set[str]) -> Label | None:
    if PNEUMONIA_FINDING in findings:
        return Label.PNEUMONIA
    if findings == {NO_FINDING}:
        return Label.NORMAL
    return None


def _map_view(value: object) -> str:
    text = str(value).strip().upper()
    return text if text in {View.PA, View.AP} else str(View.UNKNOWN)


def _map_sex(value: object) -> str:
    text = str(value).strip().upper()
    return text if text in {Sex.F, Sex.M} else str(Sex.UNKNOWN)


def _map_age(value: object) -> float | None:
    """ChestX-ray14 contains ages above 400; those rows are unusable."""
    try:
        age = float(value)
    except (TypeError, ValueError):
        return None
    return age if 0 <= age <= 120 else None
