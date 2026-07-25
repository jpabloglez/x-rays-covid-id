"""Tests for the image upload endpoint.

The point of most of these is the storage path: a request must not be able to
influence where bytes land, and must not be able to store a non-image.
"""

import hashlib
from io import BytesIO
from pathlib import Path

import pytest
from django.core.files.uploadedfile import SimpleUploadedFile
from PIL import Image

from files.views import MAX_UPLOAD_BYTES

UPLOAD_URL = "/files/"


@pytest.fixture
def media_root(settings, tmp_path):
    root = tmp_path / "media"
    root.mkdir()
    settings.MEDIA_ROOT = str(root)
    return root


def png_bytes(color="white", size=(8, 8)) -> bytes:
    buffer = BytesIO()
    Image.new("RGB", size, color).save(buffer, format="PNG")
    return buffer.getvalue()


def upload(client, payload: bytes, filename: str = "scan.png", content_type: str = "image/png"):
    return client.post(
        UPLOAD_URL,
        {"image": SimpleUploadedFile(filename, payload, content_type=content_type)},
    )


def test_accepts_a_png_and_stores_it_under_its_content_hash(client, media_root):
    payload = png_bytes()

    response = upload(client, payload)

    assert response.status_code == 201
    digest = hashlib.sha256(payload).hexdigest()
    assert response.json()["imageUrl"] == f"/media/{digest}.png"
    assert (media_root / f"{digest}.png").read_bytes() == payload


def test_traversal_filename_cannot_escape_the_media_root(client, media_root, tmp_path):
    escape_target = tmp_path / "pwned.png"

    response = upload(client, png_bytes(), filename="../../../../pwned.png")

    assert response.status_code == 201
    assert not escape_target.exists()
    # The only thing written is the hash-named file inside MEDIA_ROOT.
    written = list(media_root.iterdir())
    assert len(written) == 1
    assert written[0].name == f"{hashlib.sha256(png_bytes()).hexdigest()}.png"


def test_absolute_filename_cannot_choose_the_destination(client, media_root, tmp_path):
    response = upload(client, png_bytes(), filename=str(tmp_path / "absolute.png"))

    assert response.status_code == 201
    assert not (tmp_path / "absolute.png").exists()
    assert len(list(media_root.iterdir())) == 1


def test_rejects_a_non_image_payload(client, media_root):
    response = upload(client, b"#!/bin/sh\nrm -rf /\n", filename="payload.png")

    assert response.status_code == 400
    assert response.json()["imageUrl"] == ""
    assert list(media_root.iterdir()) == []


def test_rejects_an_image_in_an_unsupported_format(client, media_root):
    buffer = BytesIO()
    Image.new("RGB", (8, 8), "white").save(buffer, format="BMP")

    response = upload(client, buffer.getvalue(), filename="scan.bmp", content_type="image/bmp")

    assert response.status_code == 400
    assert list(media_root.iterdir()) == []


def test_rejects_an_oversized_upload(client, media_root, monkeypatch):
    monkeypatch.setattr("files.views.MAX_UPLOAD_BYTES", 128)

    response = upload(client, png_bytes(size=(256, 256)))

    assert response.status_code == 413
    assert list(media_root.iterdir()) == []


def test_the_same_image_twice_yields_one_stored_file(client, media_root):
    payload = png_bytes()

    first = upload(client, payload)
    second = upload(client, payload)

    assert first.json()["imageUrl"] == second.json()["imageUrl"]
    assert len(list(media_root.iterdir())) == 1


def test_no_partial_files_are_left_behind(client, media_root):
    upload(client, png_bytes())

    assert [path.suffix for path in Path(media_root).iterdir()] == [".png"]


def test_size_limit_is_a_sane_default():
    assert MAX_UPLOAD_BYTES == 20 * 1024 * 1024
