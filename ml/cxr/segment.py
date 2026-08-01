"""Lung segmentation for the sources that ship no masks.

COVID-19 Radiography publishes a mask per image; RSNA publishes none. Without
them the ablation speaks for 1,218 of 4,180 test images, all from one
collection, which is exactly the caveat that weakens it.

So a U-Net is trained on the masks we have and applied to the images we don't.
Two things about that are uncomfortable, and both shape the code.

**It is a domain transfer with no ground truth on the far side.** The segmenter
learns from 299x299 re-encoded PNGs and runs on 1024x1024 DICOMs decoded
through a VOI LUT. Dice can be reported on held-out COVID Radiography and
nowhere else. A number measured on the source domain is not evidence about the
target domain, and quoting it as if it were is the mistake this module exists
to avoid.

**So the far side gets a plausibility gate instead.** A lung field has
properties that hold regardless of dataset: it covers a plausible fraction of
the frame, it comes in two roughly balanced halves, and it sits in the upper
-middle of the image. Masks failing those checks are excluded from the ablation
and counted, rather than silently used. An ablation computed against a
confidently wrong mask is worse than no ablation, because it looks like a
measurement.

Training uses only the `train` split. Masks are not labels, but a segmenter
fitted to test images segments them better, and the ablation those masks feed
would inherit the optimism.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from cxr.cache import ImageCache
from cxr.preprocessing import PreprocessingSpec
from cxr.preprocessing import reference as ref

# Ranges a genuine pair of lung fields satisfies in a chest radiograph,
# whatever collection it came from. Deliberately wide: the gate is here to
# catch a segmenter that has failed outright -- an empty mask, a mask covering
# the whole frame, one lung missing -- not to enforce a house style.
MIN_AREA_FRACTION = 0.08
MAX_AREA_FRACTION = 0.60
MAX_SIDE_IMBALANCE = 0.70
MAX_VERTICAL_CENTROID = 0.80


@dataclass(frozen=True)
class Plausibility:
    """Whether a predicted mask looks like lungs, and why not if it does not."""

    area_fraction: float
    side_imbalance: float
    vertical_centroid: float
    components: int

    @property
    def failures(self) -> list[str]:
        problems = []
        if not MIN_AREA_FRACTION <= self.area_fraction <= MAX_AREA_FRACTION:
            problems.append(f"area {self.area_fraction:.3f} outside "
                            f"[{MIN_AREA_FRACTION}, {MAX_AREA_FRACTION}]")
        if self.side_imbalance > MAX_SIDE_IMBALANCE:
            problems.append(f"one side carries {self.side_imbalance:.0%} more than the other")
        if self.vertical_centroid > MAX_VERTICAL_CENTROID:
            problems.append(f"centroid sits at {self.vertical_centroid:.2f} of image height")
        return problems

    @property
    def plausible(self) -> bool:
        return not self.failures


def assess(mask: np.ndarray) -> Plausibility:
    """Shape statistics of a predicted mask, with no reference to ground truth."""
    mask = np.asarray(mask).astype(bool)
    if mask.ndim != 2:
        raise ValueError(f"expected a 2D mask, got shape {mask.shape}")

    total = float(mask.size)
    area = float(mask.sum())
    if area == 0:
        return Plausibility(0.0, 1.0, 0.0, 0)

    midpoint = mask.shape[1] // 2
    left = float(mask[:, :midpoint].sum())
    right = float(mask[:, midpoint:].sum())
    imbalance = abs(left - right) / max(left + right, 1.0)

    rows = np.argwhere(mask)[:, 0]
    return Plausibility(
        area_fraction=area / total,
        side_imbalance=imbalance,
        vertical_centroid=float(rows.mean()) / mask.shape[0],
        components=_components(mask),
    )


def _components(mask: np.ndarray, limit: int = 8) -> int:
    """Connected components, counted with a flood fill to avoid a scipy dependency.

    Capped: the exact count of a shattered mask does not matter, only that it
    shattered.
    """
    seen = np.zeros_like(mask, dtype=bool)
    found = 0
    for start in np.argwhere(mask):
        if seen[tuple(start)]:
            continue
        found += 1
        if found > limit:
            return limit
        stack = [tuple(start)]
        seen[tuple(start)] = True
        while stack:
            row, column = stack.pop()
            for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                r, c = row + dr, column + dc
                inside = 0 <= r < mask.shape[0] and 0 <= c < mask.shape[1]
                if inside and mask[r, c] and not seen[r, c]:
                    seen[r, c] = True
                    stack.append((r, c))
    return found


def dice(predicted: np.ndarray, truth: np.ndarray) -> float:
    """Dice coefficient. Only meaningful where ground truth exists."""
    predicted = np.asarray(predicted).astype(bool)
    truth = np.asarray(truth).astype(bool)
    denominator = predicted.sum() + truth.sum()
    if denominator == 0:
        return 1.0
    return float(2.0 * np.logical_and(predicted, truth).sum() / denominator)


def build_unet(spec: PreprocessingSpec):
    """A small U-Net. Segmenting two large convex blobs is not a hard problem,
    and a bigger network here would mostly memorise the source domain."""
    from monai.networks.nets import UNet

    return UNet(
        spatial_dims=2,
        in_channels=1,
        out_channels=1,
        channels=(16, 32, 64, 128),
        strides=(2, 2, 2),
        num_res_units=2,
    )


class MaskDataset:
    """Preprocessed image and its mask, both at the model's resolution.

    Reads through the same cache the classifier uses, so the segmenter sees the
    identical windowed representation. That is what makes the transfer to RSNA
    plausible at all: intensity windowing already removes most of what differs
    between a re-encoded PNG and a DICOM.
    """

    def __init__(
        self,
        frame: pd.DataFrame,
        *,
        spec: PreprocessingSpec,
        mask_roots: dict[str, Path],
        cache: ImageCache | None = None,
        image_roots: dict[str, Path] | None = None,
        augment: bool = False,
    ) -> None:
        self.frame = frame.reset_index(drop=True)
        self.spec = spec
        self.mask_roots = mask_roots
        self.cache = cache
        self.image_roots = image_roots or {}
        self.augment = augment

    def __len__(self) -> int:
        return len(self.frame)

    def __getitem__(self, position: int):
        row = self.frame.iloc[position]
        image = self._image(row)
        mask = self._mask(row)
        if self.augment:
            image, mask = _augment_pair(image, mask, seed=position)
        return image[None].astype(np.float32), mask[None].astype(np.float32)

    def _image(self, row) -> np.ndarray:
        if self.cache is not None:
            return self.cache.windowed(row["image_id"])
        from cxr.cache import deterministic_uint8

        root = self.image_roots[row["source"]]
        return deterministic_uint8(Path(root) / row["path"], self.spec) / 255.0

    def _mask(self, row) -> np.ndarray:
        from cxr.masks import load_mask

        root = self.mask_roots[row["source"]]
        mask = load_mask(Path(root) / row["mask_path"]).astype(np.float32)
        mask = ref.pad_to_square(mask, 0.0)
        return (ref.resize(mask, self.spec.target_size) > 0.5).astype(np.float32)


def _augment_pair(image: np.ndarray, mask: np.ndarray, *, seed: int):
    """Geometry applied identically to both; intensity only to the image.

    Intensity augmentation is the part that matters here. The segmenter has to
    survive a shift from re-encoded PNG to windowed DICOM, and brightness and
    gamma jitter is the cheapest way to stop it keying on the source domain's
    particular tone curve.
    """
    rng = np.random.default_rng(seed)
    if rng.random() < 0.5:
        shift = int(rng.integers(-8, 9))
        image = np.roll(image, shift, axis=1)
        mask = np.roll(mask, shift, axis=1)
    image = np.clip(image * rng.uniform(0.7, 1.3) + rng.uniform(-0.1, 0.1), 0.0, 1.0)
    image = np.power(image, rng.uniform(0.7, 1.4))
    return image.astype(np.float32), mask


@dataclass
class SegmenterReport:
    """What the segmenter is worth, stated separately per domain."""

    val_dice: float
    val_images: int
    target_images: int
    implausible: int
    reasons: dict[str, int]
    statistics: dict[str, dict[str, float]]

    def render(self) -> str:
        share = self.implausible / max(1, self.target_images)
        lines = [
            f"Dice {self.val_dice:.4f} on {self.val_images} held-out images of the "
            "source collection.",
            "",
            "That number describes the source domain only. There is no ground truth on "
            "the target collection, so its masks are judged on shape alone:",
            f"  {self.target_images - self.implausible} of {self.target_images} plausible, "
            f"{self.implausible} rejected ({share:.1%}).",
        ]
        for reason, count in sorted(self.reasons.items(), key=lambda item: -item[1]):
            lines.append(f"    {count:6d}  {reason}")
        if self.statistics:
            lines += ["", "Shape statistics per collection (mean):"]
            for source, values in self.statistics.items():
                lines.append(
                    f"  {source:<20} area {values['area_fraction']:.3f}  "
                    f"imbalance {values['side_imbalance']:.3f}  "
                    f"centroid {values['vertical_centroid']:.3f}"
                )
            lines += [
                "",
                "Compare the rows: statistics that differ markedly between collections "
                "mean the segmenter is behaving differently on the target, and the "
                "ablation built on these masks inherits that.",
            ]
        return "\n".join(lines)

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class SegmentConfig:
    """Small by design. Two convex blobs do not need a large network, and a
    bigger one would mostly memorise the source collection's tone curve."""

    epochs: int = 12
    batch_size: int = 8
    learning_rate: float = 1e-3
    workers: int = 2
    seed: int = 0
    threshold: float = 0.5


def train_segmenter(
    frame: pd.DataFrame,
    *,
    spec: PreprocessingSpec,
    mask_roots: dict[str, Path],
    cache: ImageCache | None = None,
    image_roots: dict[str, Path] | None = None,
    config: SegmentConfig | None = None,
):
    """Fit on the `train` split only, and report Dice on `val`.

    Restricting to `train` is not a formality. Masks carry no class label, but a
    segmenter that has seen the test images segments them better, and the
    ablation those masks feed would quietly inherit the optimism.
    """
    import torch
    from monai.losses import DiceLoss
    from torch.utils.data import DataLoader

    from cxr.train import resolve_device, seed_everything

    config = config or SegmentConfig()
    seed_everything(config.seed)
    device = resolve_device()

    with_masks = frame[frame["mask_path"].notna()]
    shared = {"spec": spec, "mask_roots": mask_roots, "cache": cache, "image_roots": image_roots}
    train_rows = with_masks[with_masks["split"] == "train"]
    val_rows = with_masks[with_masks["split"] == "val"]
    if train_rows.empty or val_rows.empty:
        raise ValueError("need mask-carrying rows in both the train and val splits")

    train_loader = DataLoader(
        MaskDataset(train_rows, augment=True, **shared),
        batch_size=config.batch_size, shuffle=True, num_workers=config.workers,
    )
    val_loader = DataLoader(
        MaskDataset(val_rows, **shared),
        batch_size=config.batch_size, shuffle=False, num_workers=config.workers,
    )

    model = build_unet(spec).to(device)
    loss_fn = DiceLoss(sigmoid=True)
    optimiser = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)

    best_dice, best_state = -1.0, None
    for epoch in range(config.epochs):
        model.train()
        running = 0.0
        for images, masks in train_loader:
            images, masks = images.to(device), masks.to(device)
            optimiser.zero_grad(set_to_none=True)
            loss = loss_fn(model(images), masks)
            loss.backward()
            optimiser.step()
            running += loss.detach().item()

        scores = []
        model.eval()
        with torch.no_grad():
            for images, masks in val_loader:
                predicted = torch.sigmoid(model(images.to(device))).cpu().numpy()
                for one, truth in zip(predicted, masks.numpy(), strict=True):
                    scores.append(dice(one[0] > config.threshold, truth[0] > 0.5))
        epoch_dice = float(np.mean(scores))
        print(
            f"epoch {epoch:>3}  loss {running / max(1, len(train_loader)):.4f}  "
            f"val Dice {epoch_dice:.4f}",
            flush=True,
        )
        if epoch_dice > best_dice:
            best_dice = epoch_dice
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    model.load_state_dict(best_state)
    return model, best_dice, len(val_rows)


def predict_masks(
    model,
    frame: pd.DataFrame,
    *,
    spec: PreprocessingSpec,
    out_dir: Path,
    cache: ImageCache | None = None,
    image_roots: dict[str, Path] | None = None,
    config: SegmentConfig | None = None,
) -> pd.DataFrame:
    """Write a mask per row and score its shape. Returns one row per image."""
    import torch
    from PIL import Image

    from cxr.train import resolve_device

    config = config or SegmentConfig()
    device = resolve_device()
    out_dir = Path(out_dir)
    model.eval()

    records = []
    for start in range(0, len(frame), config.batch_size):
        chunk = frame.iloc[start : start + config.batch_size]
        batch = []
        for row in chunk.to_dict("records"):
            if cache is not None:
                windowed = cache.windowed(row["image_id"])
            else:
                from cxr.cache import deterministic_uint8

                root = image_roots[row["source"]]
                windowed = deterministic_uint8(Path(root) / row["path"], spec) / 255.0
            batch.append(windowed.astype(np.float32))

        tensor = torch.from_numpy(np.stack(batch)[:, None]).to(device)
        with torch.no_grad():
            predicted = torch.sigmoid(model(tensor)).cpu().numpy()

        for row, output in zip(chunk.to_dict("records"), predicted, strict=True):
            mask = output[0] > config.threshold
            quality = assess(mask)
            relative = Path(row["path"]).with_suffix(".png")
            destination = out_dir / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            Image.fromarray((mask * 255).astype(np.uint8), mode="L").save(destination)
            records.append(
                {
                    "image_id": row["image_id"],
                    "source": row["source"],
                    "mask_path": str(relative),
                    "plausible": quality.plausible,
                    "reason": "; ".join(quality.failures),
                    **asdict(quality),
                }
            )
    return pd.DataFrame(records)


def summarise_predictions(
    predictions: pd.DataFrame, reference: pd.DataFrame | None = None
) -> dict[str, dict[str, float]]:
    """Mean shape statistics per collection, so the domains can be compared.

    This is the only handle on whether the transfer worked. A target collection
    whose masks are systematically smaller, or sit lower, is one the segmenter
    is treating differently, and the ablation built on them inherits that.
    """
    columns = ["area_fraction", "side_imbalance", "vertical_centroid"]
    frames = [predictions]
    if reference is not None and not reference.empty:
        frames.append(reference)
    combined = pd.concat(frames, ignore_index=True)
    return {
        str(source): {name: float(group[name].mean()) for name in columns}
        for source, group in combined.groupby("source")
    }
