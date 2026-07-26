"""Backbone construction, and what gets saved alongside the weights.

A checkpoint that is only weights is not reproducible. The preprocessing spec,
the class order and the gate report travel with it, because a threshold fitted
under one preprocessing and applied under another is silently wrong, and a
model card that cannot say what the corpus was measured to be is not a model
card.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from cxr.data import CLASSES
from cxr.preprocessing import PreprocessingSpec

DEFAULT_BACKBONE = "densenet121"


@dataclass(frozen=True)
class ModelConfig:
    """Everything that decides what the network is.

    DenseNet-121 is the brief's suggestion and the field's default for chest
    radiographs since CheXNet, which makes it the right Track 1 baseline: the
    point of Track 1 is to reproduce what the literature reports and then take
    it apart, not to win on architecture.
    """

    backbone: str = DEFAULT_BACKBONE
    pretrained: bool = True
    dropout: float = 0.2
    classes: tuple[str, ...] = CLASSES


def build_model(config: ModelConfig):
    """A timm backbone with a fresh head sized to `config.classes`."""
    import timm

    return timm.create_model(
        config.backbone,
        pretrained=config.pretrained,
        num_classes=len(config.classes),
        drop_rate=config.dropout,
    )


@dataclass
class Checkpoint:
    """Weights plus the context needed to use them correctly."""

    model: ModelConfig
    spec: PreprocessingSpec
    temperature: float = 1.0
    metrics: dict[str, Any] = field(default_factory=dict)
    gates: dict[str, Any] = field(default_factory=dict)
    notes: str = ""

    def write(self, directory: Path, state_dict=None) -> None:
        directory.mkdir(parents=True, exist_ok=True)
        payload = {
            "model": {**asdict(self.model), "classes": list(self.model.classes)},
            "spec": self.spec.to_dict(),
            "temperature": self.temperature,
            "metrics": self.metrics,
            "gates": self.gates,
            "notes": self.notes,
        }
        (directory / "model.json").write_text(json.dumps(payload, indent=2, default=str), "utf-8")
        self.spec.write(directory / "preprocessing.json")
        if state_dict is not None:
            import torch

            torch.save(state_dict, directory / "weights.pt")

    @classmethod
    def read(cls, directory: Path) -> Checkpoint:
        payload = json.loads((directory / "model.json").read_text(encoding="utf-8"))
        model = payload["model"]
        return cls(
            model=ModelConfig(
                backbone=model["backbone"],
                pretrained=model["pretrained"],
                dropout=model["dropout"],
                classes=tuple(model["classes"]),
            ),
            spec=PreprocessingSpec.from_dict(payload["spec"]),
            temperature=payload.get("temperature", 1.0),
            metrics=payload.get("metrics", {}),
            gates=payload.get("gates", {}),
            notes=payload.get("notes", ""),
        )
