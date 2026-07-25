"""RSNA Pneumonia Detection Challenge.

The only source in the plan shipping true DICOM, so it carries the header
handling and windowing that transfer to any real PACS-fed pipeline. Also the
only one with opacity bounding boxes, which give ground truth for the pointing
-game attribution score in Phase D.

No COVID label: released in 2018.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from cxr.manifest import Label, LabelProvenance, Sex, View
from cxr.sources.base import Source, SourceError, SourceInfo

CLASS_INFO = "stage_2_detailed_class_info.csv"
IMAGE_DIR = "stage_2_train_images"

# "No Lung Opacity / Not Normal" is deliberately unmapped: those chests are
# abnormal without an opacity, so they belong in neither class. Treating them
# as normal is the single most common way this dataset gets misused.
CLASS_MAP = {
    "Normal": Label.NORMAL,
    "Lung Opacity": Label.PNEUMONIA,
}


class RsnaPneumonia(Source):
    info = SourceInfo(
        name="rsna_pneumonia",
        title="RSNA Pneumonia Detection Challenge",
        citation="Shih et al. 2019 (RSNA / NIH)",
        licence="Kaggle competition terms; non-commercial research",
        has_covid_label=False,
        has_patient_ids=True,
        has_demographics=True,
        modality="DICOM",
        notes=(
            "Derived from ChestX-ray14, so it overlaps with it at the image "
            "level — run G2 across both before using them together. "
            "'No Lung Opacity / Not Normal' studies are excluded rather than "
            "called normal."
        ),
    )

    def build(self, root: Path) -> list[dict]:
        try:
            import pydicom
        except ImportError as error:  # pragma: no cover - depends on extras
            raise SourceError(
                "reading this source needs the imaging extra: pip install -e 'ml[imaging]'"
            ) from error

        class_info_path = root / CLASS_INFO
        if not class_info_path.exists():
            raise SourceError(f"{CLASS_INFO} not found under {root}")

        class_info = pd.read_csv(class_info_path).drop_duplicates(subset="patientId")
        if "class" not in class_info.columns:
            raise SourceError(f"{CLASS_INFO} is missing the 'class' column")
        # `class` is a keyword, so itertuples would rename it to a positional
        # attribute; renaming here keeps the loop below readable.
        class_info = class_info.rename(columns={"class": "finding"})

        image_dir = root / IMAGE_DIR
        if not image_dir.is_dir():
            raise SourceError(f"{IMAGE_DIR}/ not found under {root}")

        records: list[dict] = []
        for row in class_info.itertuples(index=False):
            label = CLASS_MAP.get(str(row.finding))
            if label is None:
                continue

            dicom_path = image_dir / f"{row.patientId}.dcm"
            if not dicom_path.exists():
                continue

            header = pydicom.dcmread(dicom_path, stop_before_pixels=True)
            relative = dicom_path.relative_to(root)
            records.append(
                {
                    "image_id": f"{self.info.name}:{row.patientId}",
                    "source": self.info.name,
                    "label": str(label),
                    "label_raw": str(row.finding),
                    "label_provenance": str(LabelProvenance.RADIOLOGIST),
                    "patient_id": f"{self.info.name}:{row.patientId}",
                    "study_id": str(getattr(header, "StudyInstanceUID", "") or "") or None,
                    "view": _map_view(getattr(header, "ViewPosition", "")),
                    "age": _map_age(getattr(header, "PatientAge", "")),
                    "sex": _map_sex(getattr(header, "PatientSex", "")),
                    "mask_path": None,
                    **self.hash_record(dicom_path, relative),
                }
            )

        if not records:
            raise SourceError(f"no usable DICOM studies found under {root}")
        return records

    def hash_record(self, absolute: Path, relative: Path) -> dict:
        """DICOM needs decoding before it can be perceptually hashed.

        The VOI LUT and PhotometricInterpretation both have to be applied
        first, or MONOCHROME1 studies hash as their own negatives and land in
        a different cluster from the identical image stored the other way up.
        """
        import numpy as np
        import pydicom
        from PIL import Image
        from pydicom.pixel_data_handlers.util import apply_voi_lut

        from cxr.hashing import dhash, sha256_file

        dataset = pydicom.dcmread(absolute)
        pixels = apply_voi_lut(dataset.pixel_array, dataset).astype(np.float32)
        if str(getattr(dataset, "PhotometricInterpretation", "")) == "MONOCHROME1":
            pixels = pixels.max() - pixels

        low, high = float(pixels.min()), float(pixels.max())
        spread = high - low
        scaled = (pixels - low) / spread * 255.0 if spread > 0 else pixels * 0.0
        image = Image.fromarray(scaled.astype("uint8"), mode="L")

        return {
            "path": str(relative),
            "sha256": sha256_file(absolute),
            "phash": dhash(image),
            "width": image.width,
            "height": image.height,
        }


def _map_view(value: object) -> str:
    text = str(value).strip().upper()
    return text if text in {View.PA, View.AP} else str(View.UNKNOWN)


def _map_sex(value: object) -> str:
    text = str(value).strip().upper()
    return text if text in {Sex.F, Sex.M} else str(Sex.UNKNOWN)


def _map_age(value: object) -> float | None:
    """DICOM ages look like '057Y'."""
    text = str(value).strip().upper().rstrip("Y")
    if not text:
        return None
    try:
        age = float(text)
    except ValueError:
        return None
    return age if 0 <= age <= 120 else None
