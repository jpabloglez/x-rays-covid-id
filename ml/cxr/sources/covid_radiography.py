"""COVID-19 Radiography Database (Qatar University / University of Dhaka).

The only source here that carries a COVID label, which makes it unavoidable
and also makes it the origin of the confound: everything it contributes is
COVID, and everything COVID comes from it.

It ships lung segmentation masks, which the in-lung attribution metric depends
on, so it earns its place despite the problems below.
"""

from __future__ import annotations

from pathlib import Path

from cxr.manifest import Label, LabelProvenance, View
from cxr.sources.base import Source, SourceError, SourceInfo

# Directory name -> canonical label. "Lung_Opacity" is deliberately absent:
# it is neither confirmed pneumonia nor normal, and folding it into either
# silently changes what the confusion matrix means. Opt in explicitly.
FOLDERS = {
    "COVID": Label.COVID,
    "Normal": Label.NORMAL,
    "Viral Pneumonia": Label.PNEUMONIA,
}

LUNG_OPACITY_FOLDER = "Lung_Opacity"


class CovidRadiography(Source):
    info = SourceInfo(
        name="covid_radiography",
        title="COVID-19 Radiography Database",
        citation="Chowdhury et al. 2020; Rahman et al. 2021",
        licence="Kaggle terms; redistribution not permitted",
        has_covid_label=True,
        has_patient_ids=False,
        has_demographics=False,
        modality="PNG",
        notes=(
            "Aggregated from many upstream repositories, so it contains "
            "near-duplicates of images in other sources. No patient ids and no "
            "demographics: G1 is blind here, and no subgroup analysis is "
            "possible on the COVID class."
        ),
    )

    def __init__(self, *, include_lung_opacity: bool = False) -> None:
        self.include_lung_opacity = include_lung_opacity

    def build(self, root: Path) -> list[dict]:
        folders = dict(FOLDERS)
        if self.include_lung_opacity:
            # Recorded as pneumonia, but label_raw keeps the distinction so the
            # decision stays visible and reversible in analysis.
            folders[LUNG_OPACITY_FOLDER] = Label.PNEUMONIA

        present = [name for name in folders if (root / name).is_dir()]
        if not present:
            raise SourceError(
                f"none of {sorted(folders)} found under {root}; expected the "
                f"COVID-19 Radiography Database layout"
            )

        records: list[dict] = []
        for folder_name in sorted(present):
            label = folders[folder_name]
            image_dir = root / folder_name / "images"
            if not image_dir.is_dir():
                image_dir = root / folder_name
            for image_path in sorted(image_dir.glob("*.png")):
                relative = image_path.relative_to(root)
                mask = root / folder_name / "masks" / image_path.name
                records.append(
                    {
                        "image_id": f"{self.info.name}:{relative.as_posix()}",
                        "source": self.info.name,
                        "label": str(label),
                        "label_raw": folder_name,
                        "label_provenance": str(LabelProvenance.FOLDER_NAME),
                        "patient_id": self.synthetic_patient_id(len(records)),
                        "study_id": None,
                        # Not recorded by this dataset. Left UNKNOWN rather
                        # than assumed PA, because assuming it would corrupt
                        # the view-position stratification everywhere else.
                        "view": str(View.UNKNOWN),
                        "age": None,
                        "sex": None,
                        "mask_path": str(mask.relative_to(root)) if mask.exists() else None,
                        **self.hash_record(image_path, relative),
                    }
                )

        if not records:
            raise SourceError(f"no PNG images found under {root}")
        return records
