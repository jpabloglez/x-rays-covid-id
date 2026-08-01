"""Freeze a checkpoint into something the web application can serve.

The serving side must not import this package's modelling code. It has no
`timm`, so it cannot rebuild a backbone from a config, and no `pandas` or
`monai`. `torch.export` solves that: the architecture travels with the weights
and loading needs nothing but torch.

**The caveats travel inside the archive.** `torch.export.save` takes extra
files and stores them in the same zip, so the gate report, the ablation
retention and the calibration temperature cannot be separated from the weights
by copying one file and forgetting the other. A served model that has lost its
confound profile is exactly the artefact this project exists to argue against,
and making the two physically inseparable is cheaper than a convention that
asks people to remember.

`torch.export` rather than `torch.jit`: the latter is deprecated in torch 2.13
and warns on every load. Both support the extra-files mechanism this relies on.

What is deliberately *not* here: preprocessing. The application imports
`cxr.preprocessing.reference` directly -- it is numpy and PIL only, so it costs
the web image nothing -- because a second implementation of the spec is how
train/serve skew starts. This project already retired the MONAI executor over a
resize divergence 229 times the quantisation bound; a third implementation in
the request path would be the same mistake with a worse blast radius.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from cxr.models import Checkpoint, build_model

METADATA_FILE = "metadata.json"
EXPORT_VERSION = 1
BATCH_DIM = "batch"


def describe(checkpoint: Checkpoint, ablation: list[dict] | None = None) -> dict[str, Any]:
    """Everything a caller needs to quote a score without misrepresenting it.

    `retention` is included at the top level rather than buried in the ablation
    list because it is the number that decides whether the score means
    anything, and a response builder reaching for one field will reach for the
    easy one.
    """
    test = checkpoint.metrics.get("test", {})
    removed = _find(ablation, "lungs removed")
    kept = _find(ablation, "lungs only")
    return {
        "export_version": EXPORT_VERSION,
        "classes": list(checkpoint.model.classes),
        "temperature": checkpoint.temperature,
        "spec": checkpoint.spec.to_dict(),
        "notes": checkpoint.notes,
        "sources": checkpoint.metrics.get("sources", []),
        "metrics": {
            "macro_auc": test.get("macro_auc"),
            "balanced_accuracy": test.get("balanced_accuracy"),
            "per_class_auc": test.get("per_class_auc", {}),
            "ece_after_calibration": checkpoint.metrics.get("ece_after_calibration"),
        },
        "gates": {
            "blocking": checkpoint.gates.get("blocking", []),
            "acknowledged": checkpoint.gates.get("acknowledged", []),
            # A skipped gate measured nothing. Carried separately from the
            # passes so a consumer counting green ticks cannot mistake one for
            # the other -- which the runner itself did until recently.
            "skipped": checkpoint.gates.get("skipped", []),
        },
        "ablation": {
            "lungs_removed_retention": removed.get("retention") if removed else None,
            "lungs_only_retention": kept.get("retention") if kept else None,
            "images": removed.get("images") if removed else None,
            "interpretation": removed.get("interpretation") if removed else None,
        },
    }


def _find(ablation: list[dict] | None, name: str) -> dict | None:
    for entry in ablation or []:
        if entry.get("name", "").strip().lower() == name:
            return entry
    return None


def export(
    checkpoint_dir: Path,
    out: Path,
    *,
    ablation: Path | None = None,
) -> dict[str, Any]:
    """Trace the model and write it with its metadata into one file."""
    import torch

    checkpoint = Checkpoint.read(checkpoint_dir)
    model = build_model(checkpoint.model)
    model.load_state_dict(torch.load(checkpoint_dir / "weights.pt", map_location="cpu"))
    model.eval()

    ablation_payload = None
    if ablation is not None:
        ablation_payload = json.loads(Path(ablation).read_text(encoding="utf-8"))
    metadata = describe(checkpoint, ablation_payload)

    spec = checkpoint.spec
    # Batch two, not one. Exporting from a single-image example makes the
    # batch dimension a constant, and the served model then refuses every
    # request that is not exactly one image -- with a shape error rather than
    # anything that reads as a capacity limit.
    example = torch.zeros(2, spec.channels, spec.target_size, spec.target_size)
    with torch.no_grad():
        program = torch.export.export(
            model, (example,), dynamic_shapes={"x": {0: torch.export.Dim(BATCH_DIM, min=1)}}
        )

    out.parent.mkdir(parents=True, exist_ok=True)
    torch.export.save(
        program, str(out), extra_files={METADATA_FILE: json.dumps(metadata, indent=2)}
    )

    _verify(out, model, example)
    return metadata


def _verify(out: Path, model, example) -> None:
    """Reload from disk and check the traced graph still agrees.

    Export silently bakes in whatever branch the example took, so an export
    that was never re-loaded and compared is an export nobody has checked. This
    costs two forward passes and turns a class of silent wrongness into a
    failure at build time.
    """
    import torch

    reloaded = torch.export.load(str(out)).module()
    with torch.no_grad():
        expected = model(example)
        actual = reloaded(example)
        # And once at a different batch size, because the whole point of the
        # dynamic dimension is that it was not baked in.
        single = reloaded(example[:1])
    if not torch.allclose(expected, actual, atol=1e-5):
        largest = float((expected - actual).abs().max())
        raise RuntimeError(
            f"exported model disagrees with the checkpoint by {largest:.2e}; "
            "the export did not capture this architecture faithfully"
        )
    if single.shape[0] != 1:
        raise RuntimeError("the exported model does not accept a single image")


def load_metadata(path: Path) -> dict[str, Any]:
    """Read the metadata back out without loading the weights."""
    import zipfile

    with zipfile.ZipFile(path) as archive:
        for name in archive.namelist():
            if name.endswith(METADATA_FILE):
                return json.loads(archive.read(name))
    raise ValueError(f"{path} carries no {METADATA_FILE}; it was not written by cxr export")


def spec_of(metadata: dict[str, Any]):
    """The preprocessing spec the model was trained under."""
    from cxr.preprocessing import PreprocessingSpec

    return PreprocessingSpec.from_dict(metadata["spec"])


__all__ = [
    "EXPORT_VERSION",
    "METADATA_FILE",
    "describe",
    "export",
    "load_metadata",
    "spec_of",
]
