"""Run the lung ablations against a trained checkpoint.

Two ablations, run together because either alone is ambiguous.

`lungs_only` keeps the lung fields and blanks the surround. `lungs_blanked`
does the opposite. Read as a pair they separate the two explanations for a high
score:

    lungs_only high, lungs_blanked low   -> the model reads the lungs
    lungs_only low,  lungs_blanked high  -> it reads everything else
    both high                            -> the signal is duplicated in both,
                                            which on this corpus means source

Reporting only `lungs_blanked` would leave the first and third cases
indistinguishable, and reporting only `lungs_only` confounds "needed the lungs"
with "needed the frame", since blanking the surround also removes collimation
edges, borders and burnt-in text.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from cxr.cache import ImageCache
from cxr.confound import Ablation, blank_lungs, coverage_note, maskable
from cxr.data import CLASSES, SplitError, class_index
from cxr.evaluate import evaluate, softmax
from cxr.masks import load_mask
from cxr.models import Checkpoint, build_model
from cxr.preprocessing import PreprocessingSpec
from cxr.preprocessing import reference as ref

BASELINE = "baseline"
LUNGS_ONLY = "lungs_only"
LUNGS_BLANKED = "lungs_blanked"
VARIANTS = (LUNGS_ONLY, LUNGS_BLANKED)


class AblationDataset:
    """The maskable rows of a split, optionally with the lungs removed or kept.

    Deliberately a separate class from RadiographDataset rather than a flag on
    it: an ablation view must never be reachable from the training path, and a
    mode argument on the shared class is exactly how that happens by accident.
    """

    def __init__(
        self,
        frame: pd.DataFrame,
        *,
        spec: PreprocessingSpec,
        variant: str,
        mask_roots: dict[str, Path],
        cache: ImageCache | None = None,
        image_roots: dict[str, Path] | None = None,
        classes: tuple[str, ...] = CLASSES,
    ) -> None:
        allowed = (BASELINE, *VARIANTS)
        if variant not in allowed:
            raise ValueError(f"unknown variant {variant!r}; expected one of {allowed}")
        if frame["mask_path"].isna().any():
            raise SplitError("every row must carry a mask; select with confound.maskable first")

        self.frame = frame.reset_index(drop=True)
        self.spec = spec
        self.variant = variant
        self.mask_roots = mask_roots
        self.cache = cache
        self.image_roots = image_roots or {}
        self.classes = classes
        self.labels = np.array(
            [class_index(classes)[label] for label in self.frame["label"]], dtype=np.int64
        )

    def __len__(self) -> int:
        return len(self.frame)

    def __getitem__(self, position: int):
        row = self.frame.iloc[position]
        if self.cache is not None:
            windowed = self.cache.windowed(row["image_id"])
        else:
            from cxr.cache import deterministic_uint8

            root = self.image_roots[row["source"]]
            windowed = deterministic_uint8(Path(root) / row["path"], self.spec) / 255.0

        if self.variant != BASELINE:
            windowed = self._apply_mask(row, windowed)

        image = ref.normalise(ref.to_channels(windowed, self.spec.channels), self.spec)
        return image, int(self.labels[position])

    def _apply_mask(self, row, windowed: np.ndarray) -> np.ndarray:
        root = self.mask_roots.get(row["source"])
        if root is None:
            raise SplitError(f"no mask root for source {row['source']!r}")

        # The mask is authored against the original image, while `windowed` has
        # been padded to square and resized. Padding first, then resampling to
        # the same side, keeps the two in register -- resampling alone would
        # shift the lungs by the padding offset on any non-square radiograph.
        mask = load_mask(Path(root) / row["mask_path"])
        mask = ref.pad_to_square(mask.astype(np.float32), 0.0)
        mask = ref.resize(mask, self.spec.target_size) > 0.5

        if self.variant == LUNGS_ONLY:
            return np.where(mask, windowed, 0.0).astype(np.float32)
        return blank_lungs(windowed, mask.astype(np.uint8))


def run(
    checkpoint_dir: Path,
    frame: pd.DataFrame,
    *,
    split: str = "test",
    mask_roots: dict[str, Path],
    cache: ImageCache | None = None,
    image_roots: dict[str, Path] | None = None,
    batch_size: int = 8,
    workers: int = 2,
) -> list[Ablation]:
    """Score the maskable rows of `split` under each ablation."""
    import torch

    from cxr.train import predict, resolve_device

    checkpoint = Checkpoint.read(checkpoint_dir)
    rows = frame[frame["split"] == split]
    subset = maskable(rows)
    if subset.empty:
        raise SplitError(f"no rows in {split!r} carry a lung mask; nothing to ablate")
    note = coverage_note(rows, subset)

    device = resolve_device()
    model = build_model(checkpoint.model).to(device)
    state = torch.load(checkpoint_dir / "weights.pt", map_location=device)
    model.load_state_dict(state)

    def score(variant: str):
        dataset = AblationDataset(
            subset, spec=checkpoint.spec, variant=variant, mask_roots=mask_roots,
            cache=cache, image_roots=image_roots, classes=checkpoint.model.classes,
        )
        loader = torch.utils.data.DataLoader(
            dataset, batch_size=batch_size, shuffle=False, num_workers=workers
        )
        logits, labels = predict(model, loader, device)
        # The trained temperature is applied throughout: an ablation compared
        # against differently-calibrated baselines would measure the
        # calibration change as well as the ablation.
        return labels, softmax(logits, checkpoint.temperature)

    baseline_labels, baseline_probabilities = score(BASELINE)
    baseline = evaluate(baseline_labels, baseline_probabilities, classes=checkpoint.model.classes)

    results = []
    for variant in VARIANTS:
        labels, probabilities = score(variant)
        blanks_lungs = variant == LUNGS_BLANKED
        results.append(
            Ablation(
                name="Lungs removed" if blanks_lungs else "Lungs only",
                images=len(subset),
                baseline=baseline,
                ablated=evaluate(labels, probabilities, classes=checkpoint.model.classes),
                removed="the lung fields" if blanks_lungs else "everything outside the lungs",
                remaining="everything outside the lungs" if blanks_lungs else "the lung fields",
                retention_is_shortcut=blanks_lungs,
                notes=note,
            )
        )
    return results
