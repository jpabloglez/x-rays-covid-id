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

    # Not just well-formed -- fetchable. A URL string and a servable file are
    # different claims, and only this checks the second one.
    assert client.get(url).status_code == 200


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


def test_the_returned_imageurl_actually_resolves(tmp_path, exported):
    """The gap the JSON-shape tests above cannot see: `imageUrl` being present
    and well-formed is not the same as the thing it points at being fetchable.
    The FastAPI migration never carried over Django's static(MEDIA_URL, ...),
    so every imageUrl this app ever returned was a dead link -- caught live, by
    opening the running app in a browser, not by any test that only checked
    the response body.
    """
    payload = _png()
    client = _client(tmp_path, model_dir=exported)
    response = client.post("/predict/", files={"image": ("c.png", payload, "image/png")})
    image_url = response.json()["imageUrl"]

    served = client.get(image_url)
    assert served.status_code == 200
    assert served.headers["content-type"] == "image/png"
    assert served.content == payload


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
    serving = tmp_path / "serving"
    serving.mkdir()
    with zipfile.ZipFile(serving / "broken.pt2", "w") as archive:
        archive.writestr("nope.txt", "not a model")

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


def test_settings_read_the_environment_when_built_not_when_imported(monkeypatch):
    """A plain dataclass default runs once, at class-definition time.

    Three fields here were plain defaults and the rest were factories, so
    `Settings()` answered from two different moments and honoured an
    environment variable or ignored it depending which field you asked for.
    """
    from api.config import Settings

    monkeypatch.setenv("MEDIA_URL", "/somewhere-else/")
    monkeypatch.setenv("MODEL_DIR", "/opt/models")
    monkeypatch.setenv("MAX_UPLOAD_BYTES", "1234")

    built = Settings()
    assert built.media_url == "/somewhere-else/"
    assert built.model_dir == "/opt/models"
    assert built.max_upload_bytes == 1234


def test_an_acknowledged_gate_reports_what_it_found(tmp_path):
    """"G5 failed" tells a reader something is wrong without telling them what.

    It also must not claim the failure was acknowledged *before training*: G5
    did not exist when Track 2 was trained and was measured against it
    afterwards, so that phrasing was simply false.
    """
    from api.inference import Prediction

    prediction = Prediction(
        track="track2", classes=["non_covid", "covid"],
        probabilities={"non_covid": 0.5, "covid": 0.5}, predicted="covid", confidence=0.5,
        metadata={
            "gates": {
                "acknowledged": ["G5"],
                "findings": {"G5": "Cramer's V 0.474 between scanner_model and class"},
            }
        },
    )
    caveats = " ".join(prediction.caveats())
    assert "Cramer's V 0.474" in caveats
    assert "before training" not in caveats


def test_an_acknowledged_gate_with_no_recorded_finding_still_says_so(tmp_path):
    from api.inference import Prediction

    prediction = Prediction(
        track="t", classes=["a", "b"], probabilities={"a": 1.0, "b": 0.0},
        predicted="a", confidence=1.0,
        metadata={"gates": {"acknowledged": ["G4"], "findings": {}}},
    )
    assert "Gate G4 fails and is acknowledged" in " ".join(prediction.caveats())


def test_models_endpoint_fills_in_gate_keys_an_older_export_omits(tmp_path, monkeypatch):
    """An export made before a field existed omits it entirely.

    Track 1 was exported before `findings` was added, so /models returned a
    gates object without it and a client following the documented shape crashed
    on `Object.entries(undefined)`. The contract belongs to this endpoint, so it
    normalises rather than passing the stored blob through.
    """
    from api import inference
    from api.config import Settings
    from fastapi.testclient import TestClient

    stale = inference.LoadedModel(
        track="ancient",
        module=None,
        spec=__import__("cxr.preprocessing", fromlist=["PreprocessingSpec"]).PreprocessingSpec(),
        metadata={"classes": ["a", "b"], "gates": {"acknowledged": ["G4"]}},
    )
    monkeypatch.setattr(inference, "registry", lambda _directory: {"ancient": stale})

    app = create_app(Settings(media_root=tmp_path / "m", model_dir=str(tmp_path)))
    gates = TestClient(app).get("/models").json()["models"][0]["gates"]

    assert gates == {
        "blocking": [],
        "acknowledged": ["G4"],
        "skipped": [],
        "findings": {},
    }


def test_a_model_exported_after_the_server_started_is_picked_up_without_a_restart(tmp_path):
    """The bug this test exists to catch: the documented workflow is to start
    the stack and *then* run `cxr export` into the mounted directory. A
    registry cached once, keyed only on the directory path, would see an
    empty directory on the very first request and serve "no models" forever
    -- the file appearing later never gets noticed, because nothing ever asks
    the directory again. It is not enough for a fresh process to find models
    that were already there; a long-running one has to notice new ones too.
    """
    pytest.importorskip("timm")
    import torch
    from cxr.export import export
    from cxr.models import Checkpoint, ModelConfig, build_model
    from cxr.preprocessing import PreprocessingSpec

    serving = tmp_path / "serving"
    client = _client(tmp_path, model_dir=serving)

    # First request: the directory does not even exist yet.
    assert client.get("/health").json()["inference_available"] is False

    # A model is exported into it after that request, exactly as `make
    # export-models` does against an already-running stack.
    spec = PreprocessingSpec(target_size=32)
    config = ModelConfig(backbone="resnet18", pretrained=False, classes=("non_covid", "covid"))
    checkpoint = Checkpoint(model=config, spec=spec, temperature=1.0)
    checkpoint_dir = tmp_path / "ckpt"
    checkpoint.write(checkpoint_dir, state_dict=build_model(config).state_dict())
    torch.save(build_model(config).state_dict(), checkpoint_dir / "weights.pt")
    export(checkpoint_dir, serving / "track2.pt2")

    # No restart, no cache-busting call -- the next request must see it.
    body = client.get("/health").json()
    assert body["inference_available"] is True
    assert body["models"] == ["track2"]


def test_an_unchanged_model_is_not_reloaded_on_every_request(tmp_path, exported, monkeypatch):
    """The other half of the fix: re-scanning the directory must not mean
    re-loading and re-tracing every model on every single request. Unchanged
    files should come back from the per-file cache."""
    from api import inference

    client = _client(tmp_path, model_dir=exported)
    client.get("/health")

    original_load = inference.load
    calls = []
    monkeypatch.setattr(
        inference, "load", lambda *a, **k: calls.append(1) or original_load(*a, **k)
    )
    client.get("/health")
    client.get("/health")
    assert calls == []


def test_a_torch_version_mismatch_names_itself_instead_of_a_bare_zipfile_error(
    tmp_path, monkeypatch
):
    """torch.export's on-disk format is not guaranteed stable across releases,
    and this project runs two different torch installs on purpose -- CUDA for
    training, CPU for serving -- so the two can drift. Reproducing a real
    cross-version failure would need two torch installs side by side; here the
    metadata read is real and only torch.export.load itself is replaced with
    something that fails the same way ("no item named 'version'").
    """
    import torch
    from api import inference

    path = tmp_path / "mismatched.pt2"
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr(
            "extra/metadata.json",
            json.dumps(
                {
                    "export_version": inference.SUPPORTED_EXPORT_VERSION,
                    "torch_version": "1.0.0",
                    "spec": {},
                    "classes": ["a", "b"],
                }
            ),
        )

    def _boom(*_args, **_kwargs):
        raise RuntimeError("no item named 'version' in the archive")

    monkeypatch.setattr(torch.export, "load", _boom)

    with pytest.raises(inference.ModelUnavailable, match=r"exported with torch 1\.0\.0"):
        inference.load(path, "track1")


def test_an_older_export_with_no_recorded_torch_version_still_gets_an_actionable_error(
    tmp_path, monkeypatch
):
    """Exports made before torch_version was recorded must not crash on a
    missing key -- they get a less specific message, not no message."""
    import torch
    from api import inference

    path = tmp_path / "no_version.pt2"
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr(
            "extra/metadata.json",
            json.dumps(
                {"export_version": inference.SUPPORTED_EXPORT_VERSION, "spec": {}, "classes": []}
            ),
        )

    monkeypatch.setattr(
        torch.export, "load", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("boom"))
    )

    with pytest.raises(inference.ModelUnavailable, match="different torch version"):
        inference.load(path, "track1")
