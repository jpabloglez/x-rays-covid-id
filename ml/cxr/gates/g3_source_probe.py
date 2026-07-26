"""G3 — how easily can a model tell the source datasets apart?

This is the diagnostic the field skipped. If a linear model on 32x32 thumbnails
can name the source repository, then source is trivially available to any
network, and any correlation between source and class is a free shortcut that
costs the model nothing to exploit.

Deliberately weak by design: logistic regression on downsampled pixels has no
capacity to learn pathology. Whatever it finds is acquisition, not disease.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from cxr.gates.base import GateResult, GateStatus

GATE = "G3"
TITLE = "Source-confound probe"

THUMBNAIL_SIDE = 32
DEFAULT_THRESHOLD = 0.75


def thumbnail_features(paths: list[Path], side: int = THUMBNAIL_SIDE) -> np.ndarray:
    """Flattened, intensity-normalised thumbnails.

    Normalising each image to zero mean and unit variance removes global
    brightness as a giveaway, so what survives is layout: borders, letterbox
    padding, burnt-in markers, the crop convention of the source.
    """
    from cxr.preprocessing.reference import load_grayscale

    rows = []
    for path in paths:
        # Via load_grayscale, not PIL: half this corpus is DICOM, and Pillow
        # cannot open it. The probe must see every source or it silently
        # scores a subset.
        source_image = Image.fromarray(load_grayscale(path), mode="F")
        grayscale = source_image.resize((side, side), Image.BILINEAR)
        pixels = np.asarray(grayscale, dtype=np.float32)
        centred = pixels - pixels.mean()
        spread = float(centred.std())
        rows.append((centred / spread if spread > 1e-6 else centred).ravel())
    return np.vstack(rows) if rows else np.empty((0, side * side), dtype=np.float32)


def resolve_paths(frame: pd.DataFrame, image_root: Path | dict[str, Path]) -> list[Path]:
    """Turn manifest paths into absolute ones.

    Sources are downloaded independently and rarely share a parent, so a single
    root cannot address a merged corpus. Passing a mapping keeps each source
    resolvable without copying tens of gigabytes into a common tree.
    """
    if not isinstance(image_root, dict):
        return [Path(image_root) / path for path in frame["path"]]

    missing = sorted(set(frame["source"].astype(str)) - set(image_root))
    if missing:
        raise ValueError(
            f"no image root given for sources {missing}; G3 needs every source "
            f"resolvable or the probe would silently score a subset"
        )
    return [
        Path(image_root[str(source)]) / path
        for source, path in zip(frame["source"], frame["path"], strict=True)
    ]


def source_confound_probe(
    frame: pd.DataFrame,
    *,
    image_root: Path | dict[str, Path] | None = None,
    features: np.ndarray | None = None,
    threshold: float = DEFAULT_THRESHOLD,
    n_splits: int = 3,
    seed: int = 0,
    feature_fn: Callable[[list[Path]], np.ndarray] = thumbnail_features,
) -> GateResult:
    """Fail if source is predictable from pixels above `threshold`.

    Scored with balanced accuracy against a chance level of 1/n_sources, and
    cross-validated grouped by patient so the probe cannot itself memorise.
    """
    sources = frame["source"].astype(str)
    distinct = sorted(set(sources))
    if len(distinct) < 2:
        return GateResult(
            gate=GATE,
            title=TITLE,
            status=GateStatus.SKIPPED,
            summary="Only one source in the manifest; there is nothing to confound with.",
        )

    if features is None:
        if image_root is None:
            raise ValueError("pass either `features` or `image_root`")
        features = feature_fn(resolve_paths(frame, image_root))

    if len(features) != len(frame):
        raise ValueError(f"features has {len(features)} rows, manifest has {len(frame)}")

    chance = 1.0 / len(distinct)
    smallest = sources.value_counts().min()
    usable_splits = min(n_splits, int(smallest), frame["patient_id"].nunique())
    if usable_splits < 2:
        return GateResult(
            gate=GATE,
            title=TITLE,
            status=GateStatus.SKIPPED,
            summary=(
                f"Not enough data to cross-validate the probe: smallest source has "
                f"{int(smallest)} images."
            ),
        )

    splitter = StratifiedGroupKFold(n_splits=usable_splits, shuffle=True, random_state=seed)
    model = make_pipeline(
        StandardScaler(),
        LogisticRegression(max_iter=2000, random_state=seed),
    )

    scores = []
    for train_positions, test_positions in splitter.split(
        features, y=sources, groups=frame["patient_id"]
    ):
        model.fit(features[train_positions], sources.iloc[train_positions])
        predicted = model.predict(features[test_positions])
        scores.append(balanced_accuracy_score(sources.iloc[test_positions], predicted))

    measured = float(np.mean(scores))
    details = {
        "sources": distinct,
        "chance_level": chance,
        "fold_scores": [round(float(score), 4) for score in scores],
        "folds": usable_splits,
    }

    if measured > threshold:
        return GateResult(
            gate=GATE,
            title=TITLE,
            status=GateStatus.FAIL,
            summary=(
                f"Source is {measured:.1%} predictable from 32x32 thumbnails alone "
                f"(chance {chance:.1%}). Provenance is free signal; any class-source "
                f"correlation will be exploited before pathology is."
            ),
            measured=measured,
            threshold=threshold,
            details=details,
        )

    return GateResult(
        gate=GATE,
        title=TITLE,
        status=GateStatus.PASS,
        summary=(
            f"Source predictable at {measured:.1%} balanced accuracy "
            f"(chance {chance:.1%}, threshold {threshold:.0%})."
        ),
        measured=measured,
        threshold=threshold,
        details=details,
    )
