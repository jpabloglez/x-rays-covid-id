"""Content-addressed upload storage.

Ported from the Django view with its security properties intact, because they
were the point of that code rather than incidental to it. The client-supplied
filename never reaches the filesystem: the name is the SHA-256 of the bytes
plus an extension chosen from a fixed map, so a request cannot pick where its
file lands or smuggle a non-image through by naming it `.png`.
"""

from __future__ import annotations

import hashlib
from io import BytesIO
from pathlib import Path

from PIL import Image as PILImage
from PIL import UnidentifiedImageError

# Pillow format name -> the extension we are willing to write for it.
ALLOWED_FORMATS = {"JPEG": ".jpg", "PNG": ".png"}


class RejectedUpload(ValueError):
    """Carries the status code, so the route does not have to infer one."""

    def __init__(self, detail: str, status: int = 400) -> None:
        super().__init__(detail)
        self.detail = detail
        self.status = status


def decode_format(payload: bytes) -> str | None:
    """The Pillow format name, or None if these bytes are not an image."""
    try:
        with PILImage.open(BytesIO(payload)) as probe:
            probe.verify()
            return probe.format
    except (UnidentifiedImageError, OSError, ValueError):
        return None


def validate(payload: bytes, *, max_bytes: int) -> str:
    """Return the extension to store under, or raise RejectedUpload."""
    if len(payload) > max_bytes:
        raise RejectedUpload(f"Image exceeds the {max_bytes // (1024 * 1024)} MB limit", 413)

    image_format = decode_format(payload)
    if image_format is None:
        raise RejectedUpload("File is not a readable image", 400)

    extension = ALLOWED_FORMATS.get(image_format)
    if extension is None:
        raise RejectedUpload(
            f"Unsupported image format {image_format}; expected JPEG or PNG", 400
        )
    return extension


def store(payload: bytes, extension: str, media_root: Path) -> str:
    """Write atomically and idempotently. Returns the stored filename."""
    filename = f"{hashlib.sha256(payload).hexdigest()}{extension}"
    media_root.mkdir(parents=True, exist_ok=True)
    destination = media_root / filename
    if destination.exists():
        # Content-addressed, so an existing file is byte-identical already.
        return filename
    staging = destination.with_name(f"{filename}.part")
    staging.write_bytes(payload)
    staging.replace(destination)
    return filename
