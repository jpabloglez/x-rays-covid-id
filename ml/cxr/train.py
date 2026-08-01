"""Track 1: fine-tune a pretrained backbone on the pooled corpus.

Track 1 trains on a corpus the gates have already condemned, deliberately. G3
measured source at 79.5% predictable and G4 found COVID drawn from exactly one
collection, so a high AUC here is the expected result and is not evidence the
model can read a chest. It is the number to hold up against Track 1's own
confound battery and against Track 2, and the gap is the finding.

Which is why the gate report is written into the checkpoint. A score reported
without it is the same score every paper in Roberts et al. (2021) reported.
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from cxr.cache import ImageCache
from cxr.data import CLASSES, class_weights, for_split
from cxr.evaluate import Evaluation, evaluate, expected_calibration_error, fit_temperature, softmax
from cxr.models import Checkpoint, ModelConfig, build_model
from cxr.preprocessing import PreprocessingSpec


@dataclass(frozen=True)
class TrainConfig:
    """Defaults sized for the machine this was developed on: a 2 GB GTX 1050.

    Batch size 16 at 320px does not fit in 2 GB, so the effective batch is made
    up with gradient accumulation instead of quietly shrinking -- optimiser
    behaviour depends on the effective batch, and a run that silently used a
    quarter of the configured one is not the run that was described.
    """

    epochs: int = 20
    batch_size: int = 8
    accumulate: int = 2
    learning_rate: float = 3e-4
    weight_decay: float = 1e-4
    warmup_epochs: int = 1
    patience: int = 5
    # How much better an epoch has to be before it counts as progress. Without
    # this, patience resets on any gain at all, and a converged run keeps going
    # on +0.0002 an epoch -- which cost the Track 1 run about two hours of
    # fourth-decimal-place noise. Set to 0.0 to restore the old behaviour.
    min_delta: float = 1e-3
    workers: int = 2
    amp: bool = True
    seed: int = 0
    freeze_backbone_epochs: int = 1

    @property
    def effective_batch(self) -> int:
        return self.batch_size * self.accumulate


def resolve_device(prefer_cuda: bool = True):
    """The device that actually works, not the one that claims to.

    `torch.cuda.is_available()` is not sufficient. A PyTorch build compiled for
    newer architectures than the installed card reports the GPU as available,
    exposes its name and compute capability, and then fails at the first kernel
    launch with `no kernel image is available for execution on the device`.
    Measured here on a GTX 1050 (sm_61) against torch 2.13+cu130, whose arch
    list starts at sm_75.

    So availability is probed rather than trusted: one tiny matmul, and a fall
    back to CPU with the reason stated. Silently training on CPU would be worse
    than either -- a run that takes ten hours instead of one, for no visible
    reason.
    """
    import torch

    if not (prefer_cuda and torch.cuda.is_available()):
        return torch.device("cpu")

    device = torch.device("cuda")
    try:
        probe = torch.zeros(8, 8, device=device)
        torch.mm(probe, probe).sum().item()
    except Exception as error:  # any failure here means "use the CPU"
        capability = torch.cuda.get_device_capability(0)
        print(
            f"CUDA reports {torch.cuda.get_device_name(0)} (sm_{capability[0]}{capability[1]}) "
            f"but cannot run a kernel on it: {type(error).__name__}. This torch build targets "
            f"{torch.cuda.get_arch_list()}. Falling back to CPU -- reinstall torch against a "
            "CUDA build that includes this card to use the GPU.",
            flush=True,
        )
        return torch.device("cpu")
    return device


def seed_everything(seed: int) -> None:
    import random

    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _loader(dataset, config: TrainConfig, *, shuffle: bool):
    import torch
    from torch.utils.data import DataLoader

    generator = torch.Generator()
    generator.manual_seed(config.seed)
    return DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=shuffle,
        num_workers=config.workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=False,
        generator=generator if shuffle else None,
        persistent_workers=config.workers > 0,
    )


def predict(model, loader, device, *, amp: bool = True) -> tuple[np.ndarray, np.ndarray]:
    """Logits and labels for a whole split, in loader order."""
    import torch

    model.eval()
    logits, labels = [], []
    with torch.no_grad():
        for images, targets in loader:
            images = images.to(device, non_blocking=True)
            with torch.autocast(device.type, enabled=amp and device.type == "cuda"):
                output = model(images)
            logits.append(output.float().cpu().numpy())
            labels.append(targets.numpy())
    return np.concatenate(logits), np.concatenate(labels)


def _provenance_note(frame: pd.DataFrame, classes: tuple[str, ...]) -> str:
    """Say what corpus this is, derived rather than asserted.

    This string used to read "Track 1: pooled corpus, knowingly confounded" no
    matter what was trained. That was true of every run that existed when it
    was written and false the moment a second task did, which is the failure
    mode this project keeps hitting: a fluent sentence describing the run its
    author had in mind rather than the one that happened.
    """
    sources = sorted(set(frame["source"]))
    task = f"{len(classes)}-class ({', '.join(classes)})"
    if len(sources) == 1:
        scope = (
            f"single source ({sources[0]}), so the class cannot be confounded with the "
            "collection -- by construction, not by measurement. Confounds within the "
            "source remain possible and are not ruled out by this."
        )
    else:
        scope = (
            f"pooled across {len(sources)} sources ({', '.join(sources)}), so the class may "
            "be confounded with the collection."
        )
    return f"{task}, {scope} Read the gate report in `gates` before quoting any number here."


def train(
    frame: pd.DataFrame,
    *,
    spec: PreprocessingSpec,
    output: Path,
    cache: ImageCache | None = None,
    image_roots: dict[str, Path] | None = None,
    model_config: ModelConfig | None = None,
    config: TrainConfig | None = None,
    gates: dict | None = None,
) -> Checkpoint:
    """Fine-tune, select on validation macro AUC, calibrate, then score test."""
    import torch
    from torch import nn

    from cxr.preprocessing.monai_pipeline import augmentation

    config = config or TrainConfig()
    model_config = model_config or ModelConfig()
    seed_everything(config.seed)

    device = resolve_device()
    output.mkdir(parents=True, exist_ok=True)
    # The model head is the single declaration of which task this is. Taking
    # the classes from anywhere else would let the labels and the output layer
    # describe different problems, and the run would train quite happily.
    classes = model_config.classes
    shared = {"spec": spec, "cache": cache, "image_roots": image_roots, "classes": classes}

    train_set = for_split(frame, "train", **shared, augment=augmentation(spec))
    val_set = for_split(frame, "val", **shared)
    train_loader = _loader(train_set, config, shuffle=True)
    val_loader = _loader(val_set, config, shuffle=False)

    model = build_model(model_config).to(device)
    weights = torch.tensor(class_weights(train_set.frame, classes=classes), device=device)
    loss_fn = nn.CrossEntropyLoss(weight=weights)
    optimiser = torch.optim.AdamW(
        model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay
    )
    steps_per_epoch = max(1, len(train_loader) // config.accumulate)
    schedule = torch.optim.lr_scheduler.OneCycleLR(
        optimiser,
        max_lr=config.learning_rate,
        total_steps=config.epochs * steps_per_epoch,
        pct_start=max(0.05, config.warmup_epochs / max(1, config.epochs)),
    )
    scaler = torch.amp.GradScaler(device.type, enabled=config.amp and device.type == "cuda")

    best_auc, best_state, best_epoch, history = -np.inf, None, -1, []
    # Tracked separately from the best score on purpose. "Which weights should
    # I keep" and "is this run still going anywhere" are different questions,
    # and answering both with one variable is what makes a converged run crawl:
    # every trivial gain both checkpoints and buys another `patience` epochs.
    progress_auc, progress_epoch = -np.inf, -1

    for epoch in range(config.epochs):
        # The head starts random against a pretrained trunk, so its first
        # gradients are large enough to damage features that took ImageNet to
        # learn. One epoch of head-only training lets it catch up first.
        frozen = epoch < config.freeze_backbone_epochs
        _set_backbone_trainable(model, not frozen)

        model.train()
        started, running = time.time(), 0.0
        optimiser.zero_grad(set_to_none=True)

        for step, (images, targets) in enumerate(train_loader):
            images = images.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            with torch.autocast(device.type, enabled=scaler.is_enabled()):
                loss = loss_fn(model(images), targets) / config.accumulate
            scaler.scale(loss).backward()
            running += loss.detach().item() * config.accumulate

            if (step + 1) % config.accumulate == 0:
                scaler.step(optimiser)
                scaler.update()
                optimiser.zero_grad(set_to_none=True)
                if schedule.last_epoch < schedule.total_steps - 1:
                    schedule.step()

        val_logits, val_labels = predict(model, val_loader, device, amp=config.amp)
        scored = evaluate(val_labels, softmax(val_logits), classes=classes)
        history.append(
            {
                "epoch": epoch,
                "train_loss": running / max(1, len(train_loader)),
                "val_macro_auc": scored.macro_auc,
                "val_balanced_accuracy": scored.balanced_accuracy,
                "seconds": round(time.time() - started, 1),
                "backbone_frozen": frozen,
            }
        )
        print(
            f"epoch {epoch:>3}  loss {history[-1]['train_loss']:.4f}  "
            f"val macro AUC {scored.macro_auc:.4f}  "
            f"balanced acc {scored.balanced_accuracy:.4f}  "
            f"{history[-1]['seconds']:.0f}s{'  [frozen]' if frozen else ''}",
            flush=True,
        )

        # Selection ignores the frozen warmup: those epochs measure the head
        # against a trunk that has not adapted, and letting one win would
        # checkpoint a model the run had not finished making.
        if not frozen and scored.macro_auc > best_auc:
            best_auc, best_epoch = scored.macro_auc, epoch
            best_state = {
                name: value.detach().cpu().clone()
                for name, value in model.state_dict().items()
            }
            # Persisted the moment it is found, not at the end of the run. An
            # epoch here costs 40 minutes, so keeping the only copy of the best
            # weights in memory means a crash at hour five loses everything.
            torch.save(
                {"epoch": epoch, "val_macro_auc": best_auc, "state_dict": best_state},
                output / "best.pt",
            )
        # Written every epoch regardless, so an interrupted run still leaves a
        # readable record of what it had reached.
        (output / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")

        # Note this is a lower bar than the checkpoint above: an epoch can be
        # worth keeping without being worth continuing for. The best weights are
        # still whatever scored highest, including a marginal winner found while
        # the run was already on the plateau.
        if not frozen and scored.macro_auc > progress_auc + config.min_delta:
            progress_auc, progress_epoch = scored.macro_auc, epoch

        # An independent check rather than the else-branch of the progress test:
        # an epoch that progresses resets progress_epoch, so the difference is
        # zero and this cannot fire on the same pass anyway.
        if progress_epoch >= 0 and epoch - progress_epoch >= config.patience:
            print(
                f"no gain above {config.min_delta:g} in {config.patience} epochs; "
                f"stopping at {epoch}. Best {best_auc:.4f} at epoch {best_epoch}.",
                flush=True,
            )
            break

    if best_state is None:
        raise RuntimeError(
            "no epoch completed unfrozen; lower freeze_backbone_epochs or raise epochs"
        )
    model.load_state_dict(best_state)

    # Calibration on `calib`, which the splitter kept clear of both training
    # and model selection precisely so this fit means something.
    calib_logits, calib_labels = predict(
        model, _loader(for_split(frame, "calib", **shared), config, shuffle=False), device,
        amp=config.amp,
    )
    temperature = fit_temperature(calib_logits, calib_labels)

    test_logits, test_labels = predict(
        model, _loader(for_split(frame, "test", **shared), config, shuffle=False), device,
        amp=config.amp,
    )
    uncalibrated = softmax(test_logits)
    calibrated = softmax(test_logits, temperature)
    result = evaluate(test_labels, calibrated, classes=classes)

    checkpoint = Checkpoint(
        model=model_config,
        spec=spec,
        temperature=temperature,
        metrics={
            "selected_epoch": best_epoch,
            "val_macro_auc": best_auc,
            "test": asdict(result),
            "ece_before_calibration": expected_calibration_error(test_labels, uncalibrated),
            "ece_after_calibration": expected_calibration_error(test_labels, calibrated),
            "history": history,
            "train_config": asdict(config),
            "effective_batch": config.effective_batch,
            "device": str(device),
            "sources": sorted(set(frame["source"])),
        },
        gates=gates or {},
        notes=_provenance_note(frame, classes),
    )
    checkpoint.write(output, state_dict=best_state)
    np.savez_compressed(
        output / "test_predictions.npz",
        logits=test_logits,
        labels=test_labels,
        image_ids=np.array(frame[frame["split"] == "test"]["image_id"].tolist()),
    )
    (output / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
    return checkpoint


def _set_backbone_trainable(model, trainable: bool) -> None:
    """Freeze everything except the classifier head."""
    import timm

    head_parameters = set()
    has_getter = hasattr(timm.models, "get_classifier")
    classifier = timm.models.get_classifier(model) if has_getter else None
    if classifier is None:
        classifier = getattr(model, "classifier", None) or getattr(model, "fc", None)
    if classifier is not None:
        head_parameters = {id(parameter) for parameter in classifier.parameters()}

    for parameter in model.parameters():
        parameter.requires_grad = trainable or id(parameter) in head_parameters


def report(checkpoint: Checkpoint) -> str:
    """The human-readable summary, with the confound stated first."""
    metrics = checkpoint.metrics
    test = metrics.get("test", {})
    lines = []
    blocking = checkpoint.gates.get("blocking") or []
    acknowledged = checkpoint.gates.get("acknowledged") or []
    if acknowledged:
        lines += [
            f"Corpus has acknowledged confounds: {', '.join(acknowledged)}. "
            "Every number below is an upper bound on what this model can do "
            "anywhere other than this corpus.",
            "",
        ]
    if blocking:
        lines += [f"WARNING: trained despite blocking gates {', '.join(blocking)}.", ""]

    lines += [
        f"selected epoch {metrics.get('selected_epoch')}  "
        f"val macro AUC {metrics.get('val_macro_auc', float('nan')):.4f}",
        f"temperature {checkpoint.temperature:.4f}  "
        f"ECE {metrics.get('ece_before_calibration', float('nan')):.4f} -> "
        f"{metrics.get('ece_after_calibration', float('nan')):.4f}",
        "",
    ]
    if test:
        lines.append(
            Evaluation(
                per_class_auc=test["per_class_auc"],
                macro_auc=test["macro_auc"],
                accuracy=test["accuracy"],
                balanced_accuracy=test["balanced_accuracy"],
                confusion=test["confusion"],
                classes=tuple(test.get("classes", CLASSES)),
            ).render()
        )
    return "\n".join(lines)
