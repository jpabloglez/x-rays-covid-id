"""What the Track 1 score is actually made of.

A macro AUC on this corpus is not a claim about reading chests, because the
gates already measured what else is available: source predictable at 79.5% from
32x32 thumbnails, and COVID drawn from exactly one collection. This module runs
the measurements that separate the two explanations.

The central one is the lung-masking ablation. Blank the lung fields and score
again: a model reading pathology should collapse towards chance, and one
reading acquisition signature -- collimation edges, burnt-in text, scanner
noise, the black border a particular pipeline leaves -- will barely notice.
DeGrave, Janizek and Lee (2021) showed published COVID classifiers doing
precisely the latter, which is why this is the number worth publishing rather
than the headline AUC.

Reported as a retention: masked AUC scaled between chance and the unmasked
score on the same images. 0.0 means the lungs carried everything; 1.0 means
they carried nothing.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from cxr.data import CLASSES
from cxr.evaluate import Evaluation, evaluate

CHANCE_AUC = 0.5


@dataclass(frozen=True)
class Ablation:
    """One ablation, always against a baseline on the identical rows.

    Comparing a masked score to the full test set's score would confound the
    ablation with the change in population, so the baseline is recomputed on
    exactly the subset that could be masked.
    """

    name: str
    images: int
    baseline: Evaluation
    ablated: Evaluation
    # What was taken away, and which direction is the good news. These are not
    # cosmetic: retention means the opposite thing depending on whether the
    # lungs were the part removed or the part kept, and a single hardcoded
    # verdict silently reports one experiment's conclusion for the other.
    removed: str = "the lung fields"
    remaining: str = "everything outside the lungs"
    retention_is_shortcut: bool = True
    notes: str = ""
    extra: dict[str, Any] = field(default_factory=dict)

    @property
    def retention(self) -> float:
        """Share of the above-chance signal that survives the ablation."""
        headroom = self.baseline.macro_auc - CHANCE_AUC
        if headroom <= 0:
            return float("nan")
        return float((self.ablated.macro_auc - CHANCE_AUC) / headroom)

    def per_class_retention(self) -> dict[str, float]:
        result = {}
        for name in self.baseline.per_class_auc:
            headroom = self.baseline.per_class_auc[name] - CHANCE_AUC
            ablated = self.ablated.per_class_auc.get(name, float("nan"))
            result[name] = (
                float("nan") if headroom <= 0 else float((ablated - CHANCE_AUC) / headroom)
            )
        return result

    def render(self) -> str:
        lines = [
            f"{self.name}  ({self.images} images)",
            f"  macro AUC {self.baseline.macro_auc:.4f} -> {self.ablated.macro_auc:.4f}"
            f"   retention {self.retention:.1%}",
        ]
        for name, value in self.per_class_retention().items():
            baseline = self.baseline.per_class_auc[name]
            ablated = self.ablated.per_class_auc.get(name, float("nan"))
            lines.append(f"    {name:<10} {baseline:.4f} -> {ablated:.4f}   {value:6.1%}")
        if self.notes:
            lines += ["", f"  {self.notes}"]
        return "\n".join(lines)


def interpret(ablation: Ablation) -> str:
    """State plainly what the retention means, so a reader cannot skip it.

    Which direction is bad depends on what was removed. With the lungs removed,
    surviving signal is signal found outside the anatomy -- a shortcut. With
    only the lungs kept, surviving signal is signal found inside the anatomy,
    which is what a credible model should have.
    """
    retention = ablation.retention
    if np.isnan(retention):
        return "The baseline is at chance on these rows; the ablation says nothing."

    share = f"{retention:.1%}"
    if not ablation.retention_is_shortcut:
        if retention >= 0.8:
            return (
                f"{share} of the signal is still available from {ablation.remaining} alone, so the "
                "anatomy does carry most of what the model could use. Read together with the "
                "complementary ablation: if that one also retains its signal, the information is "
                "duplicated inside and outside the lungs and this number proves nothing on its own."
            )
        return (
            f"only {share} of the signal survives when nothing but {ablation.remaining} is left. "
            "The model cannot do the job from the anatomy alone, which means most of what it "
            "uses lies outside it."
        )

    if retention >= 0.8:
        return (
            f"{share} of the signal survives with {ablation.removed} removed entirely. The model "
            "is almost entirely reading something other than the anatomy -- this score does not "
            "transfer to any other hospital, and must not be reported as a diagnostic result."
        )
    if retention >= 0.5:
        return (
            f"{share} of the signal survives with {ablation.removed} removed. A majority of what "
            "the model uses is outside the region the disease is in."
        )
    if retention >= 0.2:
        return (
            f"{share} survives with {ablation.removed} removed. Real anatomical signal is present, "
            "but a substantial shortcut remains and the headline number is inflated by it."
        )
    return (
        f"only {share} survives with {ablation.removed} removed, so the model is mostly reading "
        "the lung fields. That is necessary for a credible result, not sufficient -- it does not "
        "rule out a within-lung confound such as disease severity tracking view position."
    )


def maskable(frame: pd.DataFrame) -> pd.DataFrame:
    """Rows that carry a lung mask. The ablation cannot speak for the rest."""
    return frame[frame["mask_path"].notna()]


def coverage_note(frame: pd.DataFrame, subset: pd.DataFrame) -> str:
    """Say out loud which part of the test set the ablation covers."""
    missing = frame[frame["mask_path"].isna()]
    if missing.empty:
        return "Every test image carries a lung mask."
    by_source = missing["source"].value_counts().to_dict()
    covered = subset["label"].value_counts().to_dict()
    return (
        f"{len(subset)} of {len(frame)} test images carry a lung mask, covering "
        f"{covered}. Unmasked sources: {by_source}. The ablation speaks only for the "
        "covered rows, and those come from a single collection, so it does not "
        "separate anatomy from that collection's own acquisition signature."
    )


def blank_lungs(image: np.ndarray, mask: np.ndarray, fill: float = 0.0) -> np.ndarray:
    """Remove the lung fields, keeping everything else exactly as it was.

    The inverse ablation -- keeping only the lungs -- is the more obvious test
    and the weaker one: cropping to the lungs also removes the border, the
    collimation and the text, so a drop confounds "needed the lungs" with
    "needed the frame". Blanking leaves the frame intact and changes one thing.
    """
    if mask.shape != image.shape[-2:]:
        raise ValueError(f"mask {mask.shape} does not match image {image.shape[-2:]}")
    result = np.array(image, copy=True)
    result[..., mask > 0] = fill
    return result


def lung_masking_ablation(
    baseline_labels: np.ndarray,
    baseline_probabilities: np.ndarray,
    ablated_labels: np.ndarray,
    ablated_probabilities: np.ndarray,
    *,
    images: int,
    notes: str = "",
    classes: tuple[str, ...] = CLASSES,
) -> Ablation:
    """Score the same rows with and without their lung fields."""
    if not np.array_equal(baseline_labels, ablated_labels):
        raise ValueError(
            "baseline and ablated runs must cover the same rows in the same order, "
            "or the comparison measures the population change instead of the ablation"
        )
    return Ablation(
        name="Lung-masking ablation",
        images=images,
        baseline=evaluate(baseline_labels, baseline_probabilities, classes=classes),
        ablated=evaluate(ablated_labels, ablated_probabilities, classes=classes),
        notes=notes,
    )


def write_report(ablations: list[Ablation], path: Path) -> None:
    """Persist for the model card, retention included rather than derived later."""
    import json

    payload = [
        {
            "name": ablation.name,
            "images": ablation.images,
            "baseline_macro_auc": ablation.baseline.macro_auc,
            "ablated_macro_auc": ablation.ablated.macro_auc,
            "retention": ablation.retention,
            "per_class_retention": ablation.per_class_retention(),
            "interpretation": interpret(ablation),
            "notes": ablation.notes,
        }
        for ablation in ablations
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
