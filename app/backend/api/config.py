"""Runtime configuration, from the environment.

Same rule Phase A applied to the Django settings: nothing that differs between
a laptop and a deployment is written into the source, and nothing secret has a
working default.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path


def _bool(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    return default if raw is None else raw.strip().lower() in {"1", "true", "yes", "on"}


def _list(name: str, default: list[str]) -> list[str]:
    raw = os.environ.get(name)
    return default if not raw else [item.strip() for item in raw.split(",") if item.strip()]


@dataclass(frozen=True)
class Settings:
    """Every field reads the environment when an instance is built.

    `default_factory` throughout, not because three of these looked untidy as
    plain defaults but because a plain default is evaluated once, when the
    class statement runs. Half these fields were frozen at import and half were
    not, so `Settings()` answered from two different moments in time and a test
    that set an environment variable saw it honoured or ignored depending on
    which field it touched.
    """

    media_root: Path = field(
        default_factory=lambda: Path(os.environ.get("MEDIA_ROOT", "media")).resolve()
    )
    media_url: str = field(default_factory=lambda: os.environ.get("MEDIA_URL", "/media/"))
    model_dir: str = field(
        default_factory=lambda: os.environ.get("MODEL_DIR", "../../ml/models/serving")
    )
    cors_origins: list[str] = field(
        default_factory=lambda: _list("CORS_ALLOWED_ORIGINS", ["http://localhost:5173"])
    )
    debug: bool = field(default_factory=lambda: _bool("DEBUG", False))
    max_upload_bytes: int = field(
        default_factory=lambda: int(os.environ.get("MAX_UPLOAD_BYTES", 20 * 1024 * 1024))
    )


def settings() -> Settings:
    return Settings()
