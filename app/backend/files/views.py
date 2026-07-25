"""Image upload endpoint.

Uploaded files are stored under a server-generated, content-addressed name.
The client-supplied filename is never used to build a filesystem path, and the
bytes are decoded with Pillow before anything is written, so a request cannot
choose where it lands or smuggle a non-image through.
"""

import hashlib
import logging
from io import BytesIO
from pathlib import Path
from typing import ClassVar

from django.conf import settings
from PIL import Image as PILImage
from PIL import UnidentifiedImageError
from rest_framework.parsers import MultiPartParser
from rest_framework.response import Response
from rest_framework.views import APIView

from .serializers import ImageSerializer

logger = logging.getLogger(__name__)

# Pillow format name -> the extension we are willing to write for it.
ALLOWED_FORMATS = {"JPEG": ".jpg", "PNG": ".png"}

MAX_UPLOAD_BYTES = 20 * 1024 * 1024


class ImageUploadView(APIView):
    parser_classes: ClassVar[list] = [MultiPartParser]

    def post(self, request, format=None):
        serializer = ImageSerializer(data=request.data)
        if not serializer.is_valid():
            return self._failure("Image upload failed", 400)

        upload = serializer.validated_data["image"]
        if upload.size > MAX_UPLOAD_BYTES:
            return self._failure(
                f"Image exceeds the {MAX_UPLOAD_BYTES // (1024 * 1024)} MB limit", 413
            )

        upload.seek(0)
        payload = upload.read()

        image_format = self._decode_format(payload)
        if image_format is None:
            return self._failure("File is not a readable image", 400)

        extension = ALLOWED_FORMATS.get(image_format)
        if extension is None:
            return self._failure(
                f"Unsupported image format {image_format}; expected JPEG or PNG", 400
            )

        filename = f"{hashlib.sha256(payload).hexdigest()}{extension}"
        self._store(filename, payload)

        return Response(
            {
                "detail": "Image uploaded successfully",
                "imageUrl": f"{settings.MEDIA_URL}{filename}",
            },
            status=201,
        )

    @staticmethod
    def _decode_format(payload: bytes) -> str | None:
        """Return the Pillow format name, or None if the bytes are not an image."""
        try:
            with PILImage.open(BytesIO(payload)) as probe:
                probe.verify()
                return probe.format
        except (UnidentifiedImageError, OSError, ValueError):
            return None

    @staticmethod
    def _store(filename: str, payload: bytes) -> None:
        """Write the payload to MEDIA_ROOT, atomically and idempotently."""
        media_root = Path(settings.MEDIA_ROOT)
        media_root.mkdir(parents=True, exist_ok=True)
        destination = media_root / filename
        if destination.exists():
            # Content-addressed, so an existing file is byte-identical already.
            return
        staging = destination.with_name(f"{filename}.part")
        staging.write_bytes(payload)
        staging.replace(destination)

    @staticmethod
    def _failure(detail: str, status: int) -> Response:
        logger.info("Rejected upload: %s", detail)
        return Response({"detail": detail, "imageUrl": ""}, status=status)
