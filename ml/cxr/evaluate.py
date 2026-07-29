"""Evaluation: the metrics the brief asks for, plus calibration.

Per-class AUC-ROC and a confusion matrix, as specified. Accuracy is reported
but never used to select a model -- on a corpus that is 49% normal, a model
that predicts normal for everything scores 0.49 and has learned nothing.

AUC is computed one-vs-rest per class and macro-averaged. Macro rather than
weighted, because weighting by support lets the majority class carry the
number, which is the opposite of what a three-class radiograph task needs.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
from sklearn.metrics import confusion_matrix, roc_auc_score

from cxr.data import CLASSES


@dataclass(frozen=True)
class Evaluation:
    """What one split produced. Everything here goes into the model card."""

    per_class_auc: dict[str, float]
    macro_auc: float
    accuracy: float
    balanced_accuracy: float
    confusion: list[list[int]]
    classes: tuple[str, ...] = CLASSES
    extra: dict[str, Any] = field(default_factory=dict)

    def render(self) -> str:
        lines = [f"macro AUC {self.macro_auc:.4f}   accuracy {self.accuracy:.4f}   "
                 f"balanced accuracy {self.balanced_accuracy:.4f}", "", "per-class AUC:"]
        lines += [f"  {name:<10} {value:.4f}" for name, value in self.per_class_auc.items()]
        lines += ["", "confusion (rows true, columns predicted):",
                  "            " + "".join(f"{name:>12}" for name in self.classes)]
        for name, row in zip(self.classes, self.confusion, strict=True):
            lines.append(f"  {name:<10}" + "".join(f"{count:>12}" for count in row))
        return "\n".join(lines)


def evaluate(
    labels: np.ndarray, probabilities: np.ndarray, *, classes: tuple[str, ...] = CLASSES
) -> Evaluation:
    """Score predicted probabilities against integer labels."""
    if probabilities.ndim != 2 or probabilities.shape[1] != len(classes):
        raise ValueError(
            f"probabilities must be (n, {len(classes)}), got {probabilities.shape}"
        )
    if len(labels) != len(probabilities):
        raise ValueError("labels and probabilities disagree on length")

    present = sorted({int(label) for label in labels})
    per_class: dict[str, float] = {}
    for index, name in enumerate(classes):
        if index not in present or len(present) < 2:
            # A class with no positives has no ROC curve. Reporting 0.5, or
            # dropping it silently from the macro average, both invent a number.
            per_class[name] = float("nan")
            continue
        per_class[name] = float(
            roc_auc_score((labels == index).astype(int), probabilities[:, index])
        )

    scored = [value for value in per_class.values() if not np.isnan(value)]
    macro = float(np.mean(scored)) if scored else float("nan")

    predicted = probabilities.argmax(axis=1)
    matrix = confusion_matrix(labels, predicted, labels=list(range(len(classes))))
    support = matrix.sum(axis=1)
    per_class_recall = np.divide(
        matrix.diagonal(), support, out=np.full(len(classes), np.nan), where=support > 0
    )

    return Evaluation(
        per_class_auc=per_class,
        macro_auc=macro,
        accuracy=float((predicted == labels).mean()),
        balanced_accuracy=float(np.nanmean(per_class_recall)),
        confusion=matrix.tolist(),
        classes=classes,
    )


def fit_temperature(
    logits: np.ndarray, labels: np.ndarray, *, max_iterations: int = 200
) -> float:
    """Temperature scaling, fitted on the calibration split and nowhere else.

    Fitting this on `val` would fit it on data the model was already selected
    against, which is why the splitter carves `calib` out separately. Returns
    the scalar the logits should be divided by before softmax.
    """
    import torch

    if len(logits) != len(labels):
        raise ValueError("logits and labels disagree on length")

    tensor_logits = torch.as_tensor(np.asarray(logits, dtype=np.float32))
    tensor_labels = torch.as_tensor(np.asarray(labels, dtype=np.int64))
    # Optimised in log space so the temperature cannot go negative, which would
    # flip every probability rather than merely miscalibrate it.
    log_temperature = torch.zeros(1, requires_grad=True)
    optimiser = torch.optim.LBFGS([log_temperature], lr=0.1, max_iter=max_iterations)
    loss_fn = torch.nn.CrossEntropyLoss()

    def step():
        optimiser.zero_grad()
        loss = loss_fn(tensor_logits / log_temperature.exp(), tensor_labels)
        loss.backward()
        return loss

    optimiser.step(step)
    return float(log_temperature.exp().item())


def softmax(logits: np.ndarray, temperature: float = 1.0) -> np.ndarray:
    scaled = np.asarray(logits, dtype=np.float64) / temperature
    scaled -= scaled.max(axis=1, keepdims=True)
    exponentiated = np.exp(scaled)
    return exponentiated / exponentiated.sum(axis=1, keepdims=True)


def expected_calibration_error(
    labels: np.ndarray, probabilities: np.ndarray, *, bins: int = 15
) -> float:
    """How far confidence sits from accuracy, averaged over confidence bins.

    Reported because a clinician reads the number, not the ranking, and AUC is
    entirely blind to whether 0.9 means 90%.
    """
    confidence = probabilities.max(axis=1)
    correct = probabilities.argmax(axis=1) == labels
    from itertools import pairwise

    edges = np.linspace(0.0, 1.0, bins + 1)
    error = 0.0
    for low, high in pairwise(edges):
        in_bin = (confidence > low) & (confidence <= high)
        if not in_bin.any():
            continue
        error += in_bin.mean() * abs(correct[in_bin].mean() - confidence[in_bin].mean())
    return float(error)
