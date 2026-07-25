"""DICOM decoding and de-identification.

Two things here are easy to get wrong and expensive to get wrong.

MONOCHROME1 means low values are bright. Roughly a fifth of chest DICOMs in
the wild use it, and if you do not invert them they are photographic negatives
of everything else in the corpus. A network handles that trivially: it learns
which images are inverted, and since which images are inverted correlates with
the equipment that produced them, it has just learned the source.

Patient identifiers hide in more places than the obvious tags. The only safe
approach is an allowlist -- keep what is known to be needed and drop
everything else -- because a denylist is a list of the places you thought of.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np


def _apply_voi_lut(pixels, dataset):
    """pydicom moved this between 2.x and 3.x; support both."""
    try:
        from pydicom.pixels import apply_voi_lut
    except ImportError:  # pydicom < 3
        from pydicom.pixel_data_handlers.util import apply_voi_lut
    return apply_voi_lut(pixels, dataset)


# Everything needed for modelling, stratification and provenance. Anything not
# named here is dropped, including tags nobody thought to consider.
KEPT_TAGS = frozenset(
    {
        "Modality",
        "BodyPartExamined",
        "ViewPosition",
        "PatientAge",
        "PatientSex",
        "PhotometricInterpretation",
        "Rows",
        "Columns",
        "BitsAllocated",
        "BitsStored",
        "HighBit",
        "PixelRepresentation",
        "SamplesPerPixel",
        "WindowCenter",
        "WindowWidth",
        "RescaleIntercept",
        "RescaleSlope",
        "PixelSpacing",
        "ImagerPixelSpacing",
        "PresentationLUTShape",
    }
)

# Kept only because the pixel data is unreadable without them.
STRUCTURAL_TAGS = frozenset({"PixelData", "TransferSyntaxUID", "SOPClassUID"})


def decode(path: Path) -> np.ndarray:
    """Read a DICOM as a 2-D float array in display orientation.

    Applies the VOI LUT so the result matches what the radiographer saw, and
    inverts MONOCHROME1 so that in every image, bright means dense.
    """
    import pydicom

    dataset = pydicom.dcmread(path)
    pixels = _apply_voi_lut(dataset.pixel_array, dataset).astype(np.float32)

    if str(getattr(dataset, "PhotometricInterpretation", "")).strip() == "MONOCHROME1":
        pixels = pixels.max() - pixels

    if pixels.ndim == 3:
        # A colour or multi-frame study. Take the first frame's luminance
        # rather than guessing; a chest radiograph should not be either.
        pixels = pixels[..., 0] if pixels.shape[-1] <= 4 else pixels[0]

    return pixels


def is_monochrome1(path: Path) -> bool:
    import pydicom

    header = pydicom.dcmread(path, stop_before_pixels=True)
    return str(getattr(header, "PhotometricInterpretation", "")).strip() == "MONOCHROME1"


def deidentify(path: Path, destination: Path) -> dict[str, str]:
    """Write a copy carrying only allowlisted tags. Returns what was removed.

    This handles metadata only. Burnt-in text rendered into the pixels
    survives it untouched, which is why `burned_in_annotation_risk` below
    exists and why the pipeline should not treat a de-identified file as
    automatically safe to publish.
    """
    import pydicom

    dataset = pydicom.dcmread(path)
    allowed = KEPT_TAGS | STRUCTURAL_TAGS

    removed: dict[str, str] = {}
    for element in list(dataset):
        name = element.keyword or str(element.tag)
        if name not in allowed:
            removed[name] = str(element.value)[:64]
            del dataset[element.tag]

    destination.parent.mkdir(parents=True, exist_ok=True)
    dataset.save_as(destination, enforce_file_format=True)
    return removed


def burned_in_annotation_risk(path: Path) -> str | None:
    """Report what the header claims about burnt-in text.

    A 'YES' here means patient identifiers may be rendered into the pixels,
    where no tag-level de-identification can reach them. Absent the tag,
    nothing can be concluded either way -- which is not the same as safe.
    """
    import pydicom

    header = pydicom.dcmread(path, stop_before_pixels=True)
    value = getattr(header, "BurnedInAnnotation", None)
    if value is None:
        return None
    return str(value).strip().upper()
