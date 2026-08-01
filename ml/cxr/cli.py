"""Command line for assembling a corpus and gating it.

    cxr sources                             what is available and what it lacks
    cxr assemble --source NAME --root DIR   scan a download into a manifest
    cxr dedupe MANIFEST                     group near-duplicates, optionally drop
    cxr split MANIFEST                      assign patient-grouped splits
    cxr gates MANIFEST --images DIR         run the gates, exit non-zero on fail
    cxr cache MANIFEST --images DIR         precompute preprocessing once
    cxr train MANIFEST --cache DIR          fine-tune, calibrate, score

`gates` returning non-zero is the point: it is meant to sit in CI between
assembling data and training on it. `train` refuses a split whose gate report
lists blocking failures, so the two are meant to be run in that order.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

from cxr import dedupe, manifest, sources, splits
from cxr.gates import run_all
from cxr.gates.runner import BLOCKING_GATES, evaluate, render, to_json
from cxr.hashing import DEFAULT_THRESHOLD_BITS


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="cxr", description=__doc__.split("\n")[0])
    subparsers = parser.add_subparsers(dest="command", required=True)

    subparsers.add_parser("sources", help="list dataset adapters")

    assemble = subparsers.add_parser("assemble", help="build a manifest from a download")
    assemble.add_argument("--source", required=True, help=f"one of {sorted(sources.REGISTRY)}")
    assemble.add_argument("--root", required=True, type=Path, help="the downloaded directory")
    assemble.add_argument("--out", required=True, type=Path, help="manifest parquet to write")

    merge = subparsers.add_parser("merge", help="combine manifests into one corpus")
    merge.add_argument("inputs", nargs="+", type=Path)
    merge.add_argument("--out", required=True, type=Path)

    dedupe_parser = subparsers.add_parser(
        "dedupe", help="annotate near-duplicate groups, optionally dropping the extra copies"
    )
    dedupe_parser.add_argument("input", type=Path)
    dedupe_parser.add_argument("--out", required=True, type=Path)
    dedupe_parser.add_argument("--threshold", type=int, default=DEFAULT_THRESHOLD_BITS)
    dedupe_parser.add_argument(
        "--drop",
        action="store_true",
        help="keep one representative per group instead of only labelling them",
    )
    dedupe_parser.add_argument(
        "--prefer",
        default=",".join(dedupe.DEFAULT_PREFERENCE),
        help="comma-separated source order deciding which copy survives --drop",
    )

    split = subparsers.add_parser("split", help="assign patient-grouped splits")
    split.add_argument("input", type=Path)
    split.add_argument("--out", required=True, type=Path)
    split.add_argument("--holdout-source", default=None)
    split.add_argument("--folds", type=int, default=5)
    split.add_argument("--calibration-folds", type=int, default=10)
    split.add_argument("--seed", type=int, default=0)

    gates = subparsers.add_parser("gates", help="run the leakage gates")
    gates.add_argument("input", type=Path, help="a manifest carrying a split column")
    gates.add_argument(
        "--images",
        action="append",
        default=None,
        metavar="SOURCE=PATH",
        help="image root for one source, repeatable; enables G3. A bare path is "
        "accepted when the corpus has a single source.",
    )
    gates.add_argument("--json", type=Path, default=None, help="write results for the model card")
    gates.add_argument("--duplicate-threshold", type=int, default=DEFAULT_THRESHOLD_BITS)
    gates.add_argument("--source-probe-threshold", type=float, default=0.75)
    gates.add_argument("--cramers-v-threshold", type=float, default=0.40)
    gates.add_argument(
        "--acknowledge",
        default="",
        metavar="G3,G4",
        help="confound gates whose failure is expected and documented. Leakage gates "
        f"({', '.join(sorted(BLOCKING_GATES))}) always block. Naming a gate that then "
        "passes is an error, so the list cannot go stale unnoticed.",
    )

    cache_parser = subparsers.add_parser(
        "cache", help="precompute deterministic preprocessing into a memmap"
    )
    cache_parser.add_argument("input", type=Path, help="a split manifest")
    cache_parser.add_argument("--out", required=True, type=Path)
    cache_parser.add_argument("--images", action="append", metavar="SOURCE=PATH", required=True)
    cache_parser.add_argument("--target-size", type=int, default=320)
    cache_parser.add_argument("--workers", type=int, default=4)

    train_parser = subparsers.add_parser("train", help="fine-tune a backbone on a split manifest")
    train_parser.add_argument("input", type=Path, help="a split manifest")
    train_parser.add_argument("--out", required=True, type=Path)
    train_parser.add_argument("--cache", type=Path, default=None)
    train_parser.add_argument("--images", action="append", metavar="SOURCE=PATH", default=None)
    train_parser.add_argument("--gates", type=Path, default=None, help="gates.json to embed")
    train_parser.add_argument("--backbone", default="densenet121")
    train_parser.add_argument("--epochs", type=int, default=20)
    train_parser.add_argument("--batch-size", type=int, default=8)
    train_parser.add_argument("--accumulate", type=int, default=2)
    train_parser.add_argument("--learning-rate", type=float, default=3e-4)
    train_parser.add_argument("--workers", type=int, default=2)
    train_parser.add_argument("--seed", type=int, default=0)
    train_parser.add_argument("--no-amp", action="store_true")
    train_parser.add_argument("--no-pretrained", action="store_true")

    ablate_parser = subparsers.add_parser(
        "ablate", help="score a checkpoint with the lung fields removed, and kept"
    )
    ablate_parser.add_argument("input", type=Path, help="a split manifest")
    ablate_parser.add_argument("--checkpoint", required=True, type=Path)
    ablate_parser.add_argument("--masks", action="append", metavar="SOURCE=PATH", required=True)
    ablate_parser.add_argument("--cache", type=Path, default=None)
    ablate_parser.add_argument("--images", action="append", metavar="SOURCE=PATH", default=None)
    ablate_parser.add_argument("--split", default="test")
    ablate_parser.add_argument("--batch-size", type=int, default=8)
    ablate_parser.add_argument("--workers", type=int, default=2)
    ablate_parser.add_argument("--json", type=Path, default=None)

    segment_parser = subparsers.add_parser(
        "segment", help="train a lung segmenter and predict masks for sources lacking them"
    )
    segment_parser.add_argument("input", type=Path, help="a split manifest")
    segment_parser.add_argument("--out", required=True, type=Path, help="mask output directory")
    segment_parser.add_argument("--masks", action="append", metavar="SOURCE=PATH", required=True,
                                help="existing mask roots, used for training")
    segment_parser.add_argument("--cache", type=Path, default=None)
    segment_parser.add_argument("--images", action="append", metavar="SOURCE=PATH", default=None)
    segment_parser.add_argument("--manifest-out", type=Path, default=None,
                                help="manifest with predicted mask paths filled in")
    segment_parser.add_argument("--epochs", type=int, default=12)
    segment_parser.add_argument("--batch-size", type=int, default=8)
    segment_parser.add_argument("--workers", type=int, default=2)
    segment_parser.add_argument("--json", type=Path, default=None)

    args = parser.parse_args(argv)
    return {
        "sources": _sources,
        "assemble": _assemble,
        "merge": _merge,
        "dedupe": _dedupe,
        "split": _split,
        "gates": _gates,
        "cache": _cache,
        "train": _train,
        "ablate": _ablate,
        "segment": _segment,
    }[args.command](args)


def _sources(_: argparse.Namespace) -> int:
    for name in sorted(sources.REGISTRY):
        info = sources.REGISTRY[name].info
        print(f"{name}\n  {info.title} · {info.modality} · {info.licence}")
        flags = [
            f"covid label: {'yes' if info.has_covid_label else 'NO'}",
            f"patient ids: {'yes' if info.has_patient_ids else 'NO'}",
            f"demographics: {'yes' if info.has_demographics else 'NO'}",
        ]
        print(f"  {' | '.join(flags)}")
        if info.notes:
            print(f"  {info.notes}")
        print()

    capable = sources.covid_capable()
    if len(capable) <= 1:
        print(
            f"Only {capable} can supply a COVID label. Every COVID image will come from "
            f"one source, so source alone predicts that class and G4 will fail by "
            f"construction. That is a property of the available public data, not a bug."
        )
    return 0


def _assemble(args: argparse.Namespace) -> int:
    source = sources.get(args.source)
    records = source.build(args.root)
    frame = manifest.from_records(records)
    manifest.write(frame, args.out)
    print(f"{len(frame)} images from {args.source} -> {args.out}")
    print(manifest.summarise(frame).to_string())
    return 0


def _merge(args: argparse.Namespace) -> int:
    frames = [manifest.read(path) for path in args.inputs]
    combined = manifest.from_records(
        [record for frame in frames for record in frame.to_dict("records")]
    )
    manifest.write(combined, args.out)
    print(f"{len(combined)} images from {len(frames)} manifests -> {args.out}")
    print(manifest.summarise(combined).to_string())
    return 0


def _dedupe(args: argparse.Namespace) -> int:
    frame = dedupe.annotate(manifest.read(args.input), threshold=args.threshold)
    report = dedupe.summarise(frame)
    print(report.to_string(index=False) if len(report) else "no multi-image clusters")

    if args.drop:
        preference = tuple(name.strip() for name in args.prefer.split(",") if name.strip())
        frame, removed = dedupe.deduplicate(
            frame, preference=preference, threshold=args.threshold
        )
        print(f"dropped {len(removed)} redundant copies, preferring {' > '.join(preference)}")
        if len(removed):
            print(removed.groupby(["source", "label"]).size().to_string())

    frame.to_parquet(args.out, index=False)
    print(f"{len(frame)} images -> {args.out}")
    print(manifest.summarise(frame).to_string())
    return 0


def _split(args: argparse.Namespace) -> int:
    frame = manifest.read(args.input)
    config = splits.SplitConfig(
        holdout_source=args.holdout_source,
        n_folds=args.folds,
        calibration_folds=args.calibration_folds,
        seed=args.seed,
    )
    assigned = splits.assign(frame, config)
    frame.assign(split=assigned).to_parquet(args.out, index=False)
    print(splits.summarise(frame, assigned).to_string())
    return 0


def _cache(args: argparse.Namespace) -> int:
    from cxr import cache as image_cache
    from cxr.preprocessing import PreprocessingSpec

    frame = pd.read_parquet(args.input)
    roots = _image_roots(args.images)
    if not isinstance(roots, dict):
        roots = {source: Path(roots) for source in set(frame["source"])}

    spec = PreprocessingSpec(target_size=args.target_size)
    built = image_cache.build(
        frame, args.out, spec=spec, image_roots=roots, workers=args.workers
    )
    gigabytes = len(built) * spec.target_size**2 / 1e9
    print(f"cached {len(built)} images at {spec.target_size}px -> {args.out} ({gigabytes:.1f} GB)")
    return 0


def _train(args: argparse.Namespace) -> int:
    import json

    from cxr import cache as image_cache
    from cxr.models import ModelConfig
    from cxr.preprocessing import PreprocessingSpec
    from cxr.train import TrainConfig, report
    from cxr.train import train as run_training

    frame = pd.read_parquet(args.input)
    if "split" not in frame.columns:
        print(f"{args.input} has no split column; run `cxr split` first", file=sys.stderr)
        return 2

    loaded = image_cache.load(args.cache) if args.cache else None
    spec = loaded.spec if loaded else PreprocessingSpec()
    roots = _image_roots(args.images)
    if roots is not None and not isinstance(roots, dict):
        roots = {source: Path(roots) for source in set(frame["source"])}

    gates_payload = {}
    if args.gates:
        gates_payload = json.loads(Path(args.gates).read_text(encoding="utf-8"))
        if gates_payload.get("blocking"):
            print(
                f"{args.gates} reports blocking gates {gates_payload['blocking']}; "
                "fix the split rather than training on it",
                file=sys.stderr,
            )
            return 1

    checkpoint = run_training(
        frame,
        spec=spec,
        output=args.out,
        cache=loaded,
        image_roots=roots,
        model_config=ModelConfig(backbone=args.backbone, pretrained=not args.no_pretrained),
        config=TrainConfig(
            epochs=args.epochs,
            batch_size=args.batch_size,
            accumulate=args.accumulate,
            learning_rate=args.learning_rate,
            workers=args.workers,
            seed=args.seed,
            amp=not args.no_amp,
        ),
        gates=gates_payload,
    )
    print("\n" + report(checkpoint))
    print(f"\nCheckpoint written to {args.out}")
    return 0


def _ablate(args: argparse.Namespace) -> int:
    from cxr import ablate as ablation
    from cxr import cache as image_cache
    from cxr.confound import interpret, write_report

    frame = pd.read_parquet(args.input)
    loaded = image_cache.load(args.cache) if args.cache else None
    roots = _image_roots(args.images)
    if roots is not None and not isinstance(roots, dict):
        roots = {source: Path(roots) for source in set(frame["source"])}

    results = ablation.run(
        args.checkpoint,
        frame,
        split=args.split,
        mask_roots=_image_roots(args.masks),
        cache=loaded,
        image_roots=roots,
        batch_size=args.batch_size,
        workers=args.workers,
    )
    for result in results:
        print(result.render())
        print(f"\n  {interpret(result)}\n")
    if args.json:
        write_report(results, args.json)
        print(f"Written to {args.json}")
    return 0


def _segment(args: argparse.Namespace) -> int:
    import json

    from cxr import cache as image_cache
    from cxr import segment as seg
    from cxr.preprocessing import PreprocessingSpec

    frame = pd.read_parquet(args.input)
    loaded = image_cache.load(args.cache) if args.cache else None
    spec = loaded.spec if loaded else PreprocessingSpec()
    roots = _image_roots(args.images)
    if roots is not None and not isinstance(roots, dict):
        roots = {source: Path(roots) for source in set(frame["source"])}
    mask_roots = _image_roots(args.masks)

    config = seg.SegmentConfig(
        epochs=args.epochs, batch_size=args.batch_size, workers=args.workers
    )
    model, val_dice, val_images = seg.train_segmenter(
        frame, spec=spec, mask_roots=mask_roots, cache=loaded,
        image_roots=roots, config=config,
    )

    targets = frame[frame["mask_path"].isna()]
    if targets.empty:
        print("every row already carries a mask; nothing to predict")
        return 0

    predictions = seg.predict_masks(
        model, targets, spec=spec, out_dir=args.out, cache=loaded,
        image_roots=roots, config=config,
    )

    reasons: dict[str, int] = {}
    for reason in predictions.loc[~predictions["plausible"], "reason"]:
        for part in str(reason).split("; "):
            key = part.split(" ")[0] if part else "unknown"
            reasons[key] = reasons.get(key, 0) + 1

    # Shape statistics for the collection that has ground truth, so the two
    # domains can be compared rather than the target judged in isolation.
    existing = frame[frame["mask_path"].notna()]
    reference = pd.DataFrame()
    if not existing.empty:
        sample = existing.sample(min(400, len(existing)), random_state=0)
        rows = []
        for row in sample.to_dict("records"):
            from cxr.masks import load_mask
            from cxr.preprocessing import reference as ref

            mask = load_mask(Path(mask_roots[row["source"]]) / row["mask_path"]).astype("float32")
            mask = ref.resize(ref.pad_to_square(mask, 0.0), spec.target_size) > 0.5
            quality = seg.assess(mask)
            rows.append({"source": row["source"], **quality.__dict__})
        reference = pd.DataFrame(rows)

    report = seg.SegmenterReport(
        val_dice=val_dice,
        val_images=val_images,
        target_images=len(predictions),
        implausible=int((~predictions["plausible"]).sum()),
        reasons=reasons,
        statistics=seg.summarise_predictions(predictions, reference),
    )
    print()
    print(report.render())

    if args.manifest_out:
        # Only plausible masks are written back. A row left without one is
        # excluded from the ablation, which is the point: an ablation against a
        # confidently wrong mask looks like a measurement.
        usable = predictions[predictions["plausible"]].set_index("image_id")["mask_path"]
        updated = frame.copy()
        fill = updated["image_id"].map(usable)
        updated["mask_path"] = updated["mask_path"].fillna(fill)
        updated.to_parquet(args.manifest_out, index=False)
        print(f"\nManifest with predicted masks -> {args.manifest_out}")
    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(report.to_dict(), indent=2, default=str), encoding="utf-8")
        print(f"Report -> {args.json}")
    return 0


def _image_roots(values: list[str] | None) -> Path | dict[str, Path] | None:
    if not values:
        return None
    if len(values) == 1 and "=" not in values[0]:
        return Path(values[0])
    roots: dict[str, Path] = {}
    for value in values:
        source, _, path = value.partition("=")
        if not path:
            raise SystemExit(f"--images expects SOURCE=PATH, got {value!r}")
        roots[source] = Path(path)
    return roots


def _gates(args: argparse.Namespace) -> int:
    frame = pd.read_parquet(args.input)
    if "split" not in frame.columns:
        print(f"{args.input} has no split column; run `cxr split` first", file=sys.stderr)
        return 2
    assigned = frame["split"]
    manifest.validate(frame.drop(columns=["split"]))

    results = run_all(
        frame,
        assigned,
        image_root=_image_roots(args.images),
        duplicate_threshold=args.duplicate_threshold,
        source_probe_threshold=args.source_probe_threshold,
        cramers_v_threshold=args.cramers_v_threshold,
    )
    acknowledged = {name.strip() for name in args.acknowledge.split(",") if name.strip()}
    print(render(results, acknowledged))
    if args.json:
        to_json(results, args.json, acknowledged)
        print(f"\nMeasured values written to {args.json}")
    return 0 if evaluate(results, acknowledged).ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
