"""The CLI, and the end-to-end path it exists to support.

The last test here is the one that matters: assemble two sources the way the
brief describes, split them, run the gates, and confirm the pipeline refuses to
hand back a green light on a corpus where COVID comes from exactly one place.
"""

from __future__ import annotations

import json

import numpy as np
import pytest
from cxr.cli import main
from PIL import Image


def _png(path, seed=0, border=False):
    path.parent.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    pixels = (rng.random((48, 48)) * 180 + 40).astype(np.uint8)
    if border:
        pixels[:5, :] = pixels[-5:, :] = pixels[:, :5] = pixels[:, -5:] = 255
    Image.fromarray(pixels, mode="L").save(path)


def test_sources_lists_adapters(capsys):
    assert main(["sources"]) == 0
    output = capsys.readouterr().out
    assert "covid_radiography" in output
    assert "chestxray14" in output


def test_sources_lists_what_each_collection_can_and_cannot_supply(capsys):
    """The warning fires only while a single collection can supply COVID.

    With BIMCV registered that is no longer the case, so the absence of the
    warning is the assertion: it is driven by the registry rather than being a
    fixed line of text.
    """
    assert main(["sources"]) == 0
    output = capsys.readouterr().out

    assert "bimcv_covid19" in output
    assert "rsna_pneumonia" in output
    assert "covid label: NO" in output      # the pre-pandemic collections
    assert "covid label: yes" in output     # BIMCV and COVID-19 Radiography
    assert "Only [" not in output



def test_assemble_writes_a_manifest(tmp_path, capsys):
    root = tmp_path / "download"
    for index in range(4):
        _png(root / "COVID" / "images" / f"c{index}.png", seed=index)
        _png(root / "Normal" / "images" / f"n{index}.png", seed=index + 100)

    out = tmp_path / "covid.parquet"
    assert main(["assemble", "--source", "covid_radiography", "--root", str(root),
                 "--out", str(out)]) == 0
    assert out.exists()
    assert "8 images" in capsys.readouterr().out


def test_gates_refuses_a_manifest_without_splits(tmp_path, capsys):
    root = tmp_path / "download"
    _png(root / "COVID" / "images" / "c0.png")
    _png(root / "Normal" / "images" / "n0.png", seed=1)
    manifest_path = tmp_path / "m.parquet"
    main(["assemble", "--source", "covid_radiography", "--root", str(root),
          "--out", str(manifest_path)])

    assert main(["gates", str(manifest_path)]) == 2
    assert "run `cxr split` first" in capsys.readouterr().err


def test_the_briefs_configuration_fails_the_gates(tmp_path, capsys):
    """Assemble what the brief describes and watch it fail, on purpose.

    Pneumonia and normal from a pre-pandemic source, COVID from a pandemic-era
    one with its own acquisition signature. This is the corpus the dataset list
    forces, and the pipeline's job is to say so before anything is trained.
    """
    nih_root = tmp_path / "nih"
    rows = []
    for index in range(18):
        findings = "Pneumonia" if index % 2 else "No Finding"
        _png(nih_root / "images" / f"n{index}.png", seed=index)
        rows.append(
            {
                "Image Index": f"n{index}.png",
                "Finding Labels": findings,
                "Patient ID": index,
                "Patient Age": 40 + index,
                "Patient Gender": "F" if index % 2 else "M",
                "View Position": "PA",
            }
        )
    import pandas as pd

    pd.DataFrame(rows).to_csv(nih_root / "Data_Entry_2017.csv", index=False)

    covid_root = tmp_path / "covid"
    for index in range(18):
        _png(covid_root / "COVID" / "images" / f"c{index}.png", seed=500 + index, border=True)

    nih_manifest = tmp_path / "nih.parquet"
    covid_manifest = tmp_path / "covid.parquet"
    merged = tmp_path / "corpus.parquet"
    split = tmp_path / "split.parquet"
    report = tmp_path / "gates.json"

    assert main(["assemble", "--source", "chestxray14", "--root", str(nih_root),
                 "--out", str(nih_manifest)]) == 0
    assert main(["assemble", "--source", "covid_radiography", "--root", str(covid_root),
                 "--out", str(covid_manifest)]) == 0
    assert main(["merge", str(nih_manifest), str(covid_manifest), "--out", str(merged)]) == 0
    assert main(["split", str(merged), "--out", str(split), "--folds", "4",
                 "--calibration-folds", "4"]) == 0

    # Images live under two different roots here, so G3 cannot resolve them
    # from one directory; G4 alone is enough to condemn this corpus.
    exit_code = main(["gates", str(split), "--json", str(report)])
    assert exit_code == 1

    output = capsys.readouterr().out
    assert "Do not train on this split" in output

    payload = {entry["gate"]: entry for entry in json.loads(report.read_text())["gates"]}
    assert payload["G4"]["status"] == "fail"
    assert "covid" in payload["G4"]["details"]["classes_from_a_single_source"]

    # Track 1 trains on exactly this corpus on purpose, so the failure can be
    # acknowledged by name -- and the acknowledgement is recorded, not just
    # obeyed, because the model card has to say the confound was known.
    assert main(["gates", str(split), "--json", str(report), "--acknowledge", "G4"]) == 0
    assert "acknowledged as known confounds" in capsys.readouterr().out

    payload = json.loads(report.read_text())
    assert payload["acknowledged"] == ["G4"]
    assert payload["training_permitted"] is True

    # G2 passes here, so naming it is a stale acknowledgement -- an error in
    # its own right, because otherwise a fixed corpus keeps a live leakage
    # check disarmed by a line nobody revisits.
    assert main(["gates", str(split), "--acknowledge", "G2"]) == 1
    assert "G2 is acknowledged" in capsys.readouterr().out


def test_split_reports_a_summary(tmp_path, capsys):
    root = tmp_path / "download"
    for index in range(12):
        _png(root / "COVID" / "images" / f"c{index}.png", seed=index)
        _png(root / "Normal" / "images" / f"n{index}.png", seed=index + 200)
        _png(root / "Viral Pneumonia" / "images" / f"p{index}.png", seed=index + 400)

    manifest_path = tmp_path / "m.parquet"
    split = tmp_path / "s.parquet"
    main(["assemble", "--source", "covid_radiography", "--root", str(root),
          "--out", str(manifest_path)])
    capsys.readouterr()

    assert main(["split", str(manifest_path), "--out", str(split),
                 "--folds", "4", "--calibration-folds", "4"]) == 0
    output = capsys.readouterr().out
    assert "patients" in output
    assert "train" in output


def test_unknown_command_is_rejected():
    with pytest.raises(SystemExit):
        main(["nonsense"])
