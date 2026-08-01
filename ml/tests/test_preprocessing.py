"""Preprocessing: the spec, the reference executor, and DICOM handling."""

from __future__ import annotations

import numpy as np
import pytest
from cxr.preprocessing import PreprocessingSpec, ResizeMode, apply
from cxr.preprocessing import reference as ref
from cxr.preprocessing.spec import SPEC_VERSION, Windowing
from PIL import Image

pydicom = pytest.importorskip("pydicom")


# --------------------------------------------------------------------------
# Spec
# --------------------------------------------------------------------------


def test_spec_round_trips_through_json(tmp_path):
    spec = PreprocessingSpec(
        target_size=256, channels=1, normalise_mean=(0.5,), normalise_std=(0.25,)
    )
    path = tmp_path / "models" / "preprocessing.json"
    spec.write(path)
    assert PreprocessingSpec.read(path) == spec


def test_spec_rejects_a_future_version():
    """A newer release may mean something different by the same field names.

    Guessing would silently preprocess differently from how the model was
    trained, which is the exact failure this module exists to prevent.
    """
    payload = PreprocessingSpec().to_dict()
    payload["version"] = SPEC_VERSION + 1
    with pytest.raises(ValueError, match="newer release"):
        PreprocessingSpec.from_dict(payload)


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"target_size": 4}, "target_size"),
        ({"clip_percentiles": (99.0, 1.0)}, "clip_percentiles"),
        ({"clip_percentiles": (-1.0, 99.0)}, "clip_percentiles"),
        ({"channels": 2}, "channels"),
        ({"channels": 1}, "normalise_mean"),
        ({"normalise_std": (0.0, 0.1, 0.1)}, "positive"),
    ],
)
def test_spec_validates_its_fields(kwargs, message):
    with pytest.raises(ValueError, match=message):
        PreprocessingSpec(**kwargs)


# --------------------------------------------------------------------------
# Reference executor
# --------------------------------------------------------------------------


def _chest(side=96, seed=0):
    rng = np.random.default_rng(seed)
    return (rng.random((side, side)) * 200 + 20).astype(np.float32)


def test_apply_returns_the_declared_shape():
    spec = PreprocessingSpec(target_size=64, channels=3)
    assert apply(_chest(), spec).shape == (3, 64, 64)


def test_apply_supports_single_channel():
    spec = PreprocessingSpec(
        target_size=32, channels=1, normalise_mean=(0.5,), normalise_std=(0.5,)
    )
    assert apply(_chest(), spec).shape == (1, 32, 32)


def test_apply_rejects_a_colour_image():
    with pytest.raises(ValueError, match="single-channel"):
        apply(np.zeros((8, 8, 3), dtype=np.float32), PreprocessingSpec())


def test_windowing_ignores_a_single_hot_pixel():
    """A lead marker or saturated pixel must not compress the lung field."""
    image = _chest(seed=1)
    image[0, 0] = 1e6
    windowed = ref.window(image, PreprocessingSpec())
    assert windowed.max() <= 1.0
    assert windowed.std() > 0.1


def test_windowing_of_a_uniform_image_does_not_divide_by_zero():
    windowed = ref.window(np.full((16, 16), 42.0, dtype=np.float32), PreprocessingSpec())
    assert np.all(windowed == 0.5)
    assert np.isfinite(windowed).all()


def test_padding_preserves_aspect_ratio():
    """Stretching a chest to square changes the cardiothoracic ratio, and
    since source aspect ratios differ, squashing manufactures a confound."""
    tall = np.zeros((100, 50), dtype=np.float32)
    padded = ref.pad_to_square(tall, 0.0)
    assert padded.shape == (100, 100)


def test_padding_centres_the_content():
    image = np.ones((10, 4), dtype=np.float32)
    padded = ref.pad_to_square(image, 0.0)
    assert padded.shape == (10, 10)
    assert padded[:, :3].sum() == 0
    assert padded[:, 3:7].sum() == 40


def test_a_wide_and_a_tall_image_reach_the_same_shape():
    spec = PreprocessingSpec(target_size=48)
    wide = apply(np.zeros((40, 120), dtype=np.float32) + np.arange(120), spec)
    tall = apply(np.zeros((120, 40), dtype=np.float32) + np.arange(120)[:, None], spec)
    assert wide.shape == tall.shape == (3, 48, 48)


def test_normalisation_uses_the_declared_statistics():
    spec = PreprocessingSpec(
        target_size=16, channels=1, normalise_mean=(0.5,), normalise_std=(0.5,)
    )
    flat = np.full((16, 16), 100.0, dtype=np.float32)
    result = apply(flat, spec)
    # A uniform image windows to 0.5, which normalises to exactly zero.
    assert np.allclose(result, 0.0)


def test_output_is_float32_not_float64():
    """Feeding float64 to a float32 model is a silent per-batch cast."""
    assert apply(_chest(), PreprocessingSpec(target_size=32)).dtype == np.float32


def test_stretch_mode_skips_padding():
    spec = PreprocessingSpec(target_size=32, resize_mode=ResizeMode.STRETCH)
    assert apply(np.zeros((10, 80), dtype=np.float32), spec).shape == (3, 32, 32)


def test_applying_the_spec_twice_gives_the_same_answer():
    """Determinism, asserted rather than assumed. A model artifact is only
    reproducible if its preprocessing is."""
    image = _chest(seed=3)
    spec = PreprocessingSpec(target_size=64)
    assert np.array_equal(apply(image, spec), apply(image, spec))


def test_load_grayscale_reads_a_png(tmp_path):
    path = tmp_path / "chest.png"
    Image.fromarray(_chest(48).astype(np.uint8), mode="L").save(path)
    loaded = ref.load_grayscale(path)
    assert loaded.ndim == 2
    assert loaded.dtype == np.float32


# --------------------------------------------------------------------------
# DICOM
# --------------------------------------------------------------------------


def _write_dicom(path, pixels, *, photometric="MONOCHROME2", **extra):
    from pydicom.dataset import Dataset, FileMetaDataset
    from pydicom.uid import ExplicitVRLittleEndian, SecondaryCaptureImageStorage

    meta = FileMetaDataset()
    meta.MediaStorageSOPClassUID = SecondaryCaptureImageStorage
    meta.MediaStorageSOPInstanceUID = pydicom.uid.generate_uid()
    meta.TransferSyntaxUID = ExplicitVRLittleEndian

    dataset = Dataset()
    dataset.file_meta = meta
    dataset.SOPClassUID = SecondaryCaptureImageStorage
    dataset.SOPInstanceUID = meta.MediaStorageSOPInstanceUID
    dataset.Modality = "DX"
    dataset.PhotometricInterpretation = photometric
    dataset.SamplesPerPixel = 1
    dataset.BitsAllocated = 16
    dataset.BitsStored = 16
    dataset.HighBit = 15
    dataset.PixelRepresentation = 0
    dataset.Rows, dataset.Columns = pixels.shape
    dataset.PixelData = pixels.astype(np.uint16).tobytes()
    for key, value in extra.items():
        setattr(dataset, key, value)

    path.parent.mkdir(parents=True, exist_ok=True)
    dataset.save_as(path, enforce_file_format=True)
    return path


def test_monochrome1_is_inverted_to_match_everything_else(tmp_path):
    """The trap. A fifth of chest DICOMs store low-is-bright, and leaving them
    that way hands the model a perfect proxy for the equipment vendor.
    """
    ramp = np.tile(np.linspace(0, 4000, 64, dtype=np.uint16), (64, 1))

    normal = ref.load_grayscale(_write_dicom(tmp_path / "m2.dcm", ramp))
    inverted = ref.load_grayscale(
        _write_dicom(tmp_path / "m1.dcm", ramp, photometric="MONOCHROME1")
    )

    # Both must end up with dense tissue bright, so the two decode to the same
    # orientation despite being stored opposite ways.
    assert normal[0, -1] > normal[0, 0]
    assert inverted[0, 0] > inverted[0, -1]


def test_monochrome1_is_detected_from_the_header(tmp_path):
    path = _write_dicom(
        tmp_path / "m1.dcm", np.zeros((8, 8), np.uint16), photometric="MONOCHROME1"
    )
    from cxr.preprocessing.dicom import is_monochrome1

    assert is_monochrome1(path)


def test_deidentify_keeps_what_modelling_needs_and_drops_the_rest(tmp_path):
    path = _write_dicom(
        tmp_path / "in.dcm",
        np.zeros((8, 8), np.uint16),
        PatientName="DOE^JANE",
        PatientID="MRN-0001",
        PatientBirthDate="19500101",
        InstitutionName="St Elsewhere",
        ReferringPhysicianName="SMITH^JOHN",
        AccessionNumber="ACC-42",
        PatientAge="061Y",
        PatientSex="F",
        ViewPosition="PA",
    )
    from cxr.preprocessing.dicom import deidentify

    out = tmp_path / "clean" / "out.dcm"
    removed = deidentify(path, out)

    cleaned = pydicom.dcmread(out)
    assert cleaned.PatientAge == "061Y"
    assert cleaned.PatientSex == "F"
    assert cleaned.ViewPosition == "PA"
    assert cleaned.pixel_array.shape == (8, 8)

    for tag in ("PatientName", "PatientID", "PatientBirthDate", "InstitutionName",
                "ReferringPhysicianName", "AccessionNumber"):
        assert tag not in cleaned
        assert tag in removed


def test_deidentify_is_an_allowlist_not_a_denylist(tmp_path):
    """A denylist is a list of the places you thought of. An unusual private
    tag nobody enumerated must still be dropped."""
    path = _write_dicom(tmp_path / "in.dcm", np.zeros((8, 8), np.uint16))
    dataset = pydicom.dcmread(path)
    dataset.add_new(0x00091001, "LO", "SITE-PATIENT-1234")
    dataset.save_as(path, enforce_file_format=True)

    from cxr.preprocessing.dicom import deidentify

    out = tmp_path / "out.dcm"
    deidentify(path, out)
    assert 0x00091001 not in pydicom.dcmread(out)


def test_burned_in_annotation_is_reported_when_declared(tmp_path):
    from cxr.preprocessing.dicom import burned_in_annotation_risk

    flagged = _write_dicom(
        tmp_path / "y.dcm", np.zeros((8, 8), np.uint16), BurnedInAnnotation="YES"
    )
    silent = _write_dicom(tmp_path / "n.dcm", np.zeros((8, 8), np.uint16))

    assert burned_in_annotation_risk(flagged) == "YES"
    # Absent means unknown, not safe.
    assert burned_in_annotation_risk(silent) is None


def test_a_dicom_survives_the_whole_pipeline(tmp_path):
    rng = np.random.default_rng(5)
    pixels = (rng.random((80, 60)) * 3000).astype(np.uint16)
    path = _write_dicom(tmp_path / "chest.dcm", pixels, WindowCenter=1500, WindowWidth=3000)

    result = apply(ref.load_grayscale(path), PreprocessingSpec(target_size=64))
    assert result.shape == (3, 64, 64)
    assert np.isfinite(result).all()


def test_windowing_choice_is_recorded_in_the_spec():
    """Which path an image took matters: if VOI-LUT availability correlates
    with source, then so does the preprocessing, and that is a confound."""
    assert PreprocessingSpec().windowing is Windowing.VOI_LUT


def test_a_sixteen_bit_radiograph_is_not_loaded_as_a_white_rectangle(tmp_path):
    """`convert("L")` clips at 255. BIMCV's 12-bit data runs to ~4095 with its
    darkest pixel already in the hundreds, so every image would load uniform
    white -- and `window` would then return flat mid-grey without complaint.
    The model would have trained on 1463 identical squares.
    """
    from cxr.preprocessing.reference import load_grayscale

    coarse = Image.fromarray((np.random.default_rng(0).random((8, 8)) * 255).astype(np.uint8), "L")
    smooth = np.asarray(coarse.resize((64, 64), Image.BICUBIC), dtype=np.float64) / 255.0
    path = tmp_path / "chest.png"
    Image.fromarray((1000 + smooth * 3095).astype(np.uint16)).save(path)

    loaded = load_grayscale(path)
    assert loaded.max() > 255, "16-bit range was clipped away"
    assert loaded.std() > 1.0, "image arrived flat"


def test_an_eight_bit_image_still_loads_exactly_as_before(tmp_path):
    from cxr.preprocessing.reference import load_grayscale

    pixels = (np.random.default_rng(1).random((32, 32)) * 255).astype(np.uint8)
    path = tmp_path / "eight.png"
    Image.fromarray(pixels, mode="L").save(path)
    assert np.array_equal(load_grayscale(path), pixels.astype(np.float32))
