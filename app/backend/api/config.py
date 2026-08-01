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
    media_root: Path = field(
        default_factory=lambda: Path(os.environ.get("MEDIA_ROOT", "media")).resolve()
    )
    media_url: str = os.environ.get("MEDIA_URL", "/media/")
    model_dir: str = os.environ.get("MODEL_DIR", "../../ml/models/serving")
    cors_origins: list[str] = field(
        default_factory=lambda: _list("CORS_ALLOWED_ORIGINS", ["http://localhost:5173"])
    )
    debug: bool = field(default_factory=lambda: _bool("DEBUG", False))
    max_upload_bytes: int = int(os.environ.get("MAX_UPLOAD_BYTES", 20 * 1024 * 1024))


def settings() -> Settings:
    return Settings()
