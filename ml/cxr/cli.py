"""Command line for assembling a corpus and gating it.

    cxr sources                             what is available and what it lacks
    cxr assemble --source NAME --root DIR   scan a download into a manifest
    cxr split MANIFEST                      assign patient-grouped splits
    cxr gates MANIFEST --images DIR         run the gates, exit non-zero on fail

`gates` returning non-zero is the point: it is meant to sit in CI between
assembling data and training on it.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

from cxr import manifest, sources, splits
from cxr.gates import run_all
from cxr.gates.runner import any_failed, render, to_json
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

    split = subparsers.add_parser("split", help="assign patient-grouped splits")
    split.add_argument("input", type=Path)
    split.add_argument("--out", required=True, type=Path)
    split.add_argument("--holdout-source", default=None)
    split.add_argument("--folds", type=int, default=5)
    split.add_argument("--calibration-folds", type=int, default=10)
    split.add_argument("--seed", type=int, default=0)

    gates = subparsers.add_parser("gates", help="run the leakage gates")
    gates.add_argument("input", type=Path, help="a manifest carrying a split column")
    gates.add_argument("--images", type=Path, default=None, help="image root, enables G3")
    gates.add_argument("--json", type=Path, default=None, help="write results for the model card")
    gates.add_argument("--duplicate-threshold", type=int, default=DEFAULT_THRESHOLD_BITS)
    gates.add_argument("--source-probe-threshold", type=float, default=0.75)
    gates.add_argument("--cramers-v-threshold", type=float, default=0.40)

    args = parser.parse_args(argv)
    return {
        "sources": _sources,
        "assemble": _assemble,
        "merge": _merge,
        "split": _split,
        "gates": _gates,
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
        image_root=args.images,
        duplicate_threshold=args.duplicate_threshold,
        source_probe_threshold=args.source_probe_threshold,
        cramers_v_threshold=args.cramers_v_threshold,
    )
    print(render(results))
    if args.json:
        to_json(results, args.json)
        print(f"\nMeasured values written to {args.json}")
    return 1 if any_failed(results) else 0


if __name__ == "__main__":
    raise SystemExit(main())
