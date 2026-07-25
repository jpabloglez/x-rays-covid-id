"""Synthetic corpora for exercising the gates.

Real radiographs are tens of gigabytes behind registration walls, so the gates
are tested against generated images instead. That is not a compromise: it is
the only way to test a leakage detector properly, because the test needs to
*plant* the leakage and assert the gate finds it. On real data you never know
the ground truth of how confounded your corpus is — which is precisely the
situation these gates exist to escape.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pytest
from cxr import manifest
from cxr.hashing import dhash, sha256_file
from PIL import Image


@dataclass
class SourceStyle:
    """An acquisition signature: the nuisance variation a real source carries.

    `border` mimics the letterbox padding and crop conventions that differ
    between repositories; `bias` and `contrast` mimic processing pipelines.
    These are exactly the features a source probe should be able to pick up.
    """

    name: str
    border: int
    bias: float
    contrast: float


PLAIN = SourceStyle("plain", border=0, bias=0.0, contrast=1.0)
BORDERED = SourceStyle("bordered", border=6, bias=40.0, contrast=0.6)


def synth_image(rng: np.random.Generator, style: SourceStyle, side: int = 64) -> Image.Image:
    """A crude chest-like field: two darker ovals on a lighter ground.

    Anatomy is jittered per image and a few random opacities are scattered in.
    That variation is not cosmetic — without it every synthetic chest has the
    same gross structure and dHash assigns them all to one cluster, which is a
    real property of this hash on radiographs and not an artifact of the
    fixture. See the threshold note in cxr.hashing.
    """
    yy, xx = np.mgrid[0:side, 0:side]
    field = 90 + 30 * rng.random((side, side))

    # Low-frequency variation, upsampled from a coarse random field. dHash
    # compares a 9x8 downsample, so high-frequency noise averages away and only
    # structure at this scale separates one image from another — the same
    # reason positioning and exposure differences matter more to it than grain.
    # 8x8 at +/-35 puts the minimum pairwise dHash distance across a 144-image
    # corpus at 11 bits, comfortably clear of the 6-bit clustering threshold.
    # Weaker variation than this and unrelated patients start colliding.
    coarse = rng.uniform(-35, 35, size=(8, 8))
    field += np.asarray(
        Image.fromarray(coarse).resize((side, side), Image.BICUBIC), dtype=np.float64
    )

    for side_sign in (-1, 1):
        centre_x = side * (0.5 + side_sign * rng.uniform(0.13, 0.22))
        centre_y = side * rng.uniform(0.42, 0.58)
        width = side * rng.uniform(0.12, 0.20)
        height = side * rng.uniform(0.22, 0.32)
        lung = ((xx - centre_x) / width) ** 2 + ((yy - centre_y) / height) ** 2
        field -= rng.uniform(30, 60) * np.exp(-lung)

    for _ in range(rng.integers(2, 6)):
        blob_x, blob_y = rng.uniform(0.15, 0.85, size=2) * side
        radius = side * rng.uniform(0.04, 0.11)
        blob = ((xx - blob_x) / radius) ** 2 + ((yy - blob_y) / radius) ** 2
        field += rng.uniform(-35, 35) * np.exp(-blob)

    field = field * style.contrast + style.bias
    if style.border:
        field[: style.border, :] = 255
        field[-style.border :, :] = 255
        field[:, : style.border] = 255
        field[:, -style.border :] = 255
    return Image.fromarray(np.clip(field, 0, 255).astype(np.uint8), mode="L")


def write_corpus(
    root: Path,
    spec: list[tuple[str, SourceStyle, str, int]],
    *,
    seed: int = 0,
    images_per_patient: int = 1,
) -> list[dict]:
    """Materialise images and return manifest records.

    `spec` is a list of (source, style, label, patient_count).
    """
    rng = np.random.default_rng(seed)
    records: list[dict] = []
    # Patients are numbered per source and never reused across labels — a
    # patient has one diagnosis here, and reusing ids would quietly hand the
    # splitter an impossible grouping.
    next_patient: dict[str, int] = {}
    for source, style, label, patient_count in spec:
        for _ in range(patient_count):
            patient_index = next_patient.get(source, 0)
            next_patient[source] = patient_index + 1
            patient_id = f"{source}:p{patient_index:04d}"
            for image_index in range(images_per_patient):
                relative = Path(source) / label / f"p{patient_index:04d}_{image_index}.png"
                absolute = root / relative
                absolute.parent.mkdir(parents=True, exist_ok=True)
                image = synth_image(rng, style)
                image.save(absolute)
                records.append(
                    {
                        "image_id": f"{source}:{relative.as_posix()}",
                        "source": source,
                        "path": str(relative),
                        "label": label,
                        "label_raw": label,
                        "label_provenance": "folder_name",
                        "patient_id": patient_id,
                        "study_id": f"{patient_id}:s{image_index}",
                        "view": "PA",
                        "age": 40.0 + (patient_index % 40),
                        "sex": "F" if patient_index % 2 else "M",
                        "sha256": sha256_file(absolute),
                        "phash": dhash(image),
                        "width": image.width,
                        "height": image.height,
                    }
                )
    return records


@pytest.fixture
def clean_corpus(tmp_path: Path) -> tuple[Path, object]:
    """Two sources, every class present in both, no duplicates.

    Two images per patient, because patients really do come back for a second
    film and that is the case G1 exists to catch — a corpus with one image per
    patient cannot distinguish grouped splitting from ungrouped splitting.
    """
    spec = [
        ("alpha", PLAIN, "normal", 12),
        ("alpha", PLAIN, "pneumonia", 12),
        ("alpha", PLAIN, "covid", 12),
        ("beta", PLAIN, "normal", 12),
        ("beta", PLAIN, "pneumonia", 12),
        ("beta", PLAIN, "covid", 12),
    ]
    records = write_corpus(tmp_path, spec, seed=1, images_per_patient=2)
    return tmp_path, manifest.from_records(records)


@pytest.fixture
def confounded_corpus(tmp_path: Path) -> tuple[Path, object]:
    """The corpus the brief produces: COVID from its own source, visibly styled.

    Pneumonia and normal come from a pre-pandemic collection; COVID can only
    come from somewhere else, and that somewhere else has its own borders and
    processing. Class is now readable from provenance.
    """
    spec = [
        ("chestxray14", PLAIN, "normal", 14),
        ("chestxray14", PLAIN, "pneumonia", 14),
        ("covid_radiography", BORDERED, "covid", 14),
    ]
    records = write_corpus(tmp_path, spec, seed=2)
    return tmp_path, manifest.from_records(records)
