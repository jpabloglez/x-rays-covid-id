"""The HTTP surface, and the caveats it must not be able to drop.

The upload tests are ported from the Django suite deliberately: the security
properties there were the point of that code, and a migration that quietly lost
one of them would be a regression nobody notices until it matters.
"""

from __future__ import annotations

import io
import json
import zipfile

import numpy as np
import pytest
from PIL import Image

pytest.importorskip("fastapi")
pytest.importorskip("torch")

from api.config import Settings
from api.main import create_app


def _png(seed=0, size=(64, 64)) -> bytes:
    rng = np.random.default_rng(seed)
    pixels = (rng.random(size) * 255).astype(np.uint8)
    buffer = io.BytesIO()
    Image.fromarray(pixels, mode="L").save(buffer, format="PNG")
    return buffer.getvalue()


def _client(tmp_path, model_dir=None):
    from fastapi.testclient import TestClient

    return TestClient(
        create_app(
            Settings(
                media_root=tmp_path / "media",
                media_url="/media/",
                model_dir=str(model_dir or tmp_path / "no-models"),
                cors_origins=["http://localhost:5173"],
            )
        )
    )


@pytest.fixture
def exported(tmp_path):
    """A real export, produced the way the CLI produces one."""
    pytest.importorskip("timm")
    import torch
    from cxr.export import export
    from cxr.models import Checkpoint, ModelConfig, build_model
    from cxr.preprocessing import PreprocessingSpec

    spec = PreprocessingSpec(target_size=32)
    config = ModelConfig(backbone="resnet18", pretrained=False, classes=("non_covid", "covid"))
    checkpoint = Checkpoint(
        model=config,
        spec=spec,
        temperature=2.0,
        metrics={"test": {"macro_auc": 0.746}, "sources": ["bimcv_covid19"]},
        gates={"blocking": [], "acknowledged": [], "skipped": ["G4"]},
    )
    directory = tmp_path / "ckpt"
    checkpoint.write(directory, state_dict=build_model(config).state_dict())
    torch.save(build_model(config).state_dict(), directory / "weights.pt")

    ablation = tmp_path / "ablation.json"
    ablation.write_text(json.dumps([
        {"name": "Lungs removed", "retention": 0.837, "images": 292, "interpretation": "..."},
    ]))

    serving = tmp_path / "serving"
    export(directory, serving / "track2.pt2", ablation=ablation)
    # Clear the process-wide cache so each test sees its own directory.
    from api import inference

    inference.registry.cache_clear()
    return serving


# --------------------------------------------------------------- upload parity


def test_a_valid_png_is_stored_under_a_content_addressed_name(tmp_path):
    client = _client(tmp_path)
    response = client.post("/files/", files={"image": ("chest.png", _png(), "image/png")})

    assert response.status_code == 201
    url = response.json()["imageUrl"]
    assert url.startswith("/media/") and url.endswith(".png")
    # The client's filename never reaches the filesystem.
    assert "chest" not in url


def test_the_same_bytes_land_on_the_same_name_twice(tmp_path):
    client = _client(tmp_path)
    first = client.post("/files/", files={"image": ("a.png", _png(1), "image/png")})
    second = client.post("/files/", files={"image": ("b.png", _png(1), "image/png")})
    assert first.json()["imageUrl"] == second.json()["imageUrl"]


def test_a_file_that_is_not_an_image_is_refused(tmp_path):
    client = _client(tmp_path)
    response = client.post("/files/", files={"image": ("x.png", b"not an image", "image/png")})
    assert response.status_code == 400
    assert "not a readable image" in response.json()["detail"]


def test_an_oversized_upload_is_refused_with_413(tmp_path):
    from fastapi.testclient import TestClient

    app = create_app(Settings(media_root=tmp_path / "m", model_dir=str(tmp_path / "none"),
                              max_upload_bytes=128))
    response = TestClient(app).post("/files/", files={"image": ("big.png", _png(), "image/png")})
    assert response.status_code == 413


def test_a_gif_is_refused_even_though_pillow_reads_it(tmp_path):
    """Readable is not the same as allowed. The extension written to disk comes
    from a fixed map, so a format outside it has nowhere to go."""
    buffer = io.BytesIO()
    Image.fromarray(np.zeros((8, 8), np.uint8), mode="L").save(buffer, format="GIF")
    client = _client(tmp_path)
    response = client.post("/files/", files={"image": ("x.gif", buffer.getvalue(), "image/gif")})
    assert response.status_code == 400
    assert "GIF" in response.json()["detail"]


# ------------------------------------------------------------------ inference


def test_health_reports_when_no_models_are_loaded(tmp_path):
    """The service starts without exports and says so, rather than refusing to
    boot -- an operator can then see the problem without reading logs."""
    body = _client(tmp_path).get("/health").json()
    assert body["status"] == "ok"
    assert body["inference_available"] is False


def test_predict_returns_503_when_nothing_is_loaded(tmp_path):
    client = _client(tmp_path)
    response = client.post("/predict/", files={"image": ("c.png", _png(), "image/png")})
    assert response.status_code == 503
    assert "cxr export" in response.json()["detail"]


def test_predict_scores_the_image_and_attaches_the_evidence(tmp_path, exported):
    client = _client(tmp_path, model_dir=exported)
    response = client.post("/predict/", files={"image": ("c.png", _png(), "image/png")})

    assert response.status_code == 200
    body = response.json()
    assert len(body["predictions"]) == 1
    prediction = body["predictions"][0]

    assert set(prediction["probabilities"]) == {"non_covid", "covid"}
    assert prediction["probabilities"]["covid"] + prediction["probabilities"]["non_covid"] == (
        pytest.approx(1.0)
    )
    assert prediction["lungs_removed_retention"] == pytest.approx(0.837)
    assert prediction["macro_auc"] == pytest.approx(0.746)


def test_a_high_retention_is_stated_in_words_not_just_a_number(tmp_path, exported):
    """A client rendering the probability and skipping the float still has to
    walk past a sentence saying the number is mostly not anatomy."""
    client = _client(tmp_path, model_dir=exported)
    body = client.post("/predict/", files={"image": ("c.png", _png(), "image/png")}).json()
    caveats = " ".join(body["predictions"][0]["caveats"])
    assert "not the anatomy" in caveats


def test_skipped_gates_are_reported_as_unmeasured(tmp_path, exported):
    client = _client(tmp_path, model_dir=exported)
    body = client.post("/predict/", files={"image": ("c.png", _png(), "image/png")}).json()
    caveats = " ".join(body["predictions"][0]["caveats"])
    assert "G4" in caveats
    assert "a skipped gate is not a passed gate" in caveats


def test_every_prediction_says_the_image_is_out_of_distribution(tmp_path, exported):
    """The caveat that applies to literally every request. A quoted AUC
    describes held-out images from the training collections; an upload is from
    somewhere else by definition."""
    client = _client(tmp_path, model_dir=exported)
    body = client.post("/predict/", files={"image": ("c.png", _png(), "image/png")}).json()
    caveats = " ".join(body["predictions"][0]["caveats"])
    assert "do not describe the accuracy of this prediction" in caveats


def test_the_response_carries_a_disclaimer(tmp_path, exported):
    client = _client(tmp_path, model_dir=exported)
    body = client.post("/predict/", files={"image": ("c.png", _png(), "image/png")}).json()
    assert "not a diagnostic device" in body["disclaimer"]


def test_a_single_model_says_there_is_nothing_to_compare(tmp_path, exported):
    """Rather than implying a consensus that one model cannot provide."""
    client = _client(tmp_path, model_dir=exported)
    body = client.post("/predict/", files={"image": ("c.png", _png(), "image/png")}).json()
    assert "nothing to compare" in body["comparison"]


def test_models_endpoint_describes_without_an_image(tmp_path, exported):
    body = _client(tmp_path, model_dir=exported).get("/models").json()
    assert body["models"][0]["track"] == "track2"
    assert body["models"][0]["gates"]["skipped"] == ["G4"]


def test_a_corrupt_export_is_skipped_rather_than_taking_the_service_down(tmp_path):
    """One bad file must not stop the others being served."""
    from api import inference

    serving = tmp_path / "serving"
    serving.mkdir()
    with zipfile.ZipFile(serving / "broken.pt2", "w") as archive:
        archive.writestr("nope.txt", "not a model")

    inference.registry.cache_clear()
    body = _client(tmp_path, model_dir=serving).get("/health").json()
    assert body["status"] == "ok"
    assert body["models"] == []


def test_preprocessing_matches_the_training_reference(tmp_path):
    """The whole reason the service imports cxr rather than reimplementing.

    If this ever fails, the image the model is served differs from the image it
    was trained on, which is the failure the two-executor design was retired to
    avoid.
    """
    from api.inference import prepare
    from cxr.preprocessing import PreprocessingSpec
    from cxr.preprocessing import reference as ref

    spec = PreprocessingSpec(target_size=32)
    payload = _png(7, size=(80, 50))
    image = Image.open(io.BytesIO(payload))

    served = prepare(image, spec)
    trained = ref.apply(np.asarray(image.convert("L"), dtype=np.float32), spec)
    assert np.array_equal(served, trained)
