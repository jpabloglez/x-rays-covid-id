"""Loading the exported models and scoring an image with them.

Two things here are deliberate and worth defending.

**Preprocessing is imported, never reimplemented.** `cxr.preprocessing.reference`
is numpy and PIL only -- no torch, timm, monai or pandas -- so importing it
costs this image nothing and guarantees the bytes fed to the model at request
time are the bytes it was trained on. The alternative, reimplementing the spec
from `preprocessing.json`, is how train/serve skew starts: the research package
already retired its MONAI executor over a resize that diverged from the
reference by 229 times the quantisation bound. A third implementation sitting
in the request path would be the same mistake with a worse blast radius.

**The two models answer different questions.** Track 1 is three-class over a
corpus where COVID came from a single collection; Track 2 is two-class within
one hospital network. Presenting them as one question in two flavours would
misrepresent both, so a prediction carries its own class set, its own gate
report and its own ablation retention, and the response says what the pair is
for rather than averaging them into a number.
"""

from __future__ import annotations

import json
import logging
import zipfile
from dataclasses import dataclass
from functools import lru_cache
from io import BytesIO
from pathlib import Path
from typing import Any

import numpy as np
from cxr.preprocessing import PreprocessingSpec
from cxr.preprocessing import reference as ref
from PIL import Image

logger = logging.getLogger(__name__)

METADATA_FILE = "metadata.json"
SUPPORTED_EXPORT_VERSION = 1


class ModelUnavailable(RuntimeError):
    """No usable export on disk. The API stays up and says so."""


@dataclass(frozen=True)
class Prediction:
    """One model's answer, inseparable from what that answer is worth."""

    track: str
    classes: list[str]
    probabilities: dict[str, float]
    predicted: str
    confidence: float
    metadata: dict[str, Any]

    @property
    def retention(self) -> float | None:
        return self.metadata.get("ablation", {}).get("lungs_removed_retention")

    def caveats(self) -> list[str]:
        """Stated per response, not left to a footnote somebody removes.

        Derived from the metadata rather than written out per track, so a model
        retrained with different properties cannot keep quoting the old ones.
        """
        notes: list[str] = []
        retention = self.retention
        if retention is None:
            notes.append(
                "No lung ablation was run for this model, so there is no evidence about "
                "whether it reads the anatomy or the acquisition."
            )
        elif retention >= 0.8:
            notes.append(
                f"{retention:.0%} of this model's discriminative signal survives having the "
                "lung fields blanked out, so most of what it uses is not the anatomy."
            )
        elif retention >= 0.5:
            notes.append(
                f"{retention:.0%} of this model's signal survives with the lung fields "
                "removed; a majority of what it uses lies outside them."
            )
        else:
            notes.append(
                f"{retention:.0%} of this model's signal survives with the lung fields "
                "removed, which rules out one shortcut but not every shortcut."
            )

        gates = self.metadata.get("gates", {})
        if gates.get("acknowledged"):
            notes.append(
                f"Gates {', '.join(gates['acknowledged'])} failed and were acknowledged as "
                "known confounds before training."
            )
        if gates.get("skipped"):
            notes.append(
                f"Gates {', '.join(gates['skipped'])} did not run, so nothing was measured "
                "for them -- a skipped gate is not a passed gate."
            )
        sources = self.metadata.get("sources") or []
        if len(sources) == 1:
            notes.append(
                f"Trained on a single collection ({sources[0]}), so class cannot be "
                "confounded with provenance -- by construction, not by measurement."
            )
        elif len(sources) > 1:
            notes.append(
                f"Trained across {len(sources)} collections ({', '.join(sources)}), so the "
                "class may be predictable from provenance alone."
            )

        # The caveat that applies to every request and is easiest to forget.
        # The quoted AUC describes held-out images from the collections the
        # model was trained on. An uploaded radiograph is from somewhere else
        # by definition, and a model shown to read acquisition signature rather
        # than anatomy is precisely the one whose score will not carry over.
        where = f"held-out images from {', '.join(sources)}" if sources else "its own test split"
        notes.append(
            f"Reported performance was measured on {where}. This image did not come from "
            "there, so the quoted figures do not describe the accuracy of this prediction."
        )
        return notes


@dataclass
class LoadedModel:
    track: str
    module: Any
    spec: PreprocessingSpec
    metadata: dict[str, Any]

    @property
    def classes(self) -> list[str]:
        return list(self.metadata["classes"])

    @property
    def temperature(self) -> float:
        return float(self.metadata.get("temperature", 1.0))


def read_metadata(path: Path) -> dict[str, Any]:
    with zipfile.ZipFile(path) as archive:
        for name in archive.namelist():
            if name.endswith(METADATA_FILE):
                return json.loads(archive.read(name))
    raise ModelUnavailable(
        f"{path.name} carries no {METADATA_FILE}; it was not written by cxr export"
    )


def load(path: Path, track: str) -> LoadedModel:
    import torch

    metadata = read_metadata(path)
    version = metadata.get("export_version")
    if version != SUPPORTED_EXPORT_VERSION:
        # Refused rather than attempted. A newer export may have moved a field
        # this code reads for the caveats, and a response that silently drops
        # its confound profile is worse than one that never arrives.
        raise ModelUnavailable(
            f"{path.name} is export version {version}; this service supports "
            f"{SUPPORTED_EXPORT_VERSION}. Re-export it or upgrade the service."
        )
    module = torch.export.load(str(path)).module()
    return LoadedModel(
        track=track,
        module=module,
        spec=PreprocessingSpec.from_dict(metadata["spec"]),
        metadata=metadata,
    )


@lru_cache(maxsize=1)
def registry(directory: str) -> dict[str, LoadedModel]:
    """Every export in `directory`, keyed by filename stem.

    Cached because loading is seconds and the files do not change under a
    running process. A missing directory is not an error here -- the service
    starts, reports no models, and the endpoints say so.
    """
    root = Path(directory)
    if not root.is_dir():
        logger.warning("No model directory at %s; /predict will report no models", root)
        return {}

    models: dict[str, LoadedModel] = {}
    for path in sorted(root.glob("*.pt2")):
        try:
            models[path.stem] = load(path, path.stem)
        except (ModelUnavailable, OSError, KeyError, ValueError) as error:
            # One bad export must not take down the others; the service reports
            # what it has rather than failing to start.
            logger.error("Skipping %s: %s", path.name, error)
    if not models:
        logger.warning("No usable exports in %s", root)
    return models


def decode(payload: bytes) -> Image.Image:
    """Bytes to a PIL image, or a ValueError naming the problem."""
    try:
        image = Image.open(BytesIO(payload))
        image.load()
    except Exception as error:  # Pillow raises a wide family here
        raise ValueError("file is not a readable image") from error
    return image


def prepare(image: Image.Image, spec: PreprocessingSpec) -> np.ndarray:
    """Run the reference preprocessing, exactly as training did.

    Goes through `ref.apply`, which is the same function the cache builder
    calls. The one difference from training is that this starts from decoded
    bytes rather than a path, so the DICOM branch of `load_grayscale` is not
    reachable here -- uploads are PNG and JPEG.
    """
    grayscale = np.asarray(image.convert("L"), dtype=np.float32)
    return ref.apply(grayscale, spec)


def softmax(logits: np.ndarray, temperature: float = 1.0) -> np.ndarray:
    scaled = logits / max(temperature, 1e-6)
    shifted = scaled - scaled.max(axis=-1, keepdims=True)
    exponentiated = np.exp(shifted)
    return exponentiated / exponentiated.sum(axis=-1, keepdims=True)


def predict(model: LoadedModel, image: Image.Image) -> Prediction:
    import torch

    tensor = prepare(image, model.spec)[None, ...]
    with torch.no_grad():
        logits = model.module(torch.from_numpy(tensor)).numpy()

    # Temperature from the checkpoint, fitted on a split held out from both
    # training and model selection. Skipping it would leave the confident-and-
    # wrong outputs that make an uncalibrated model dangerous to show a user.
    probabilities = softmax(logits, model.temperature)[0]
    classes = model.classes
    order = int(np.argmax(probabilities))
    return Prediction(
        track=model.track,
        classes=classes,
        probabilities={
            name: float(value)
            for name, value in zip(classes, probabilities, strict=True)
        },
        predicted=classes[order],
        confidence=float(probabilities[order]),
        metadata=model.metadata,
    )
