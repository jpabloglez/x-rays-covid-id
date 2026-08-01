"""The manifest: one row per image, and the only thing downstream code reads.

Every gate, split and training run is a function of this table. That makes its
invariants worth enforcing hard at assembly time — a malformed row does not
crash later, it silently weakens a gate, which is the failure mode this whole
package exists to prevent.
"""

from __future__ import annotations

from enum import StrEnum
from pathlib import Path

import pandas as pd


class Label(StrEnum):
    """The classes the project targets, across two different tasks.

    Track 1 pools collections and uses {normal, pneumonia, covid}. Track 2
    discriminates within a single collection and uses {covid, non_covid},
    because a molecularly negative patient in a pandemic-era hospital series is
    usually not healthy -- in BIMCV only a fifth of the negative partition is
    radiologically normal, the rest carrying effusion, pneumonia, cardiomegaly
    and so on. Calling those `normal` would be the same error this project
    refuses for RSNA's "No Lung Opacity / Not Normal", and it would turn Track 2
    into sick-versus-healthy, reproducing the confound it exists to escape.

    NON_COVID therefore means "tested, not COVID", not "nothing wrong".
    """

    NORMAL = "normal"
    PNEUMONIA = "pneumonia"
    COVID = "covid"
    NON_COVID = "non_covid"


class LabelProvenance(StrEnum):
    """How a label was arrived at. These are not equivalent evidence.

    Results on the PCR-confirmed subset and on folder-name labels belong in
    different rows of the model card, never averaged together.
    """

    PCR = "pcr"
    RADIOLOGIST = "radiologist"
    NLP_MINED = "nlp_mined"
    FOLDER_NAME = "folder_name"


class View(StrEnum):
    """Projection. AP is the confounder: portable films are taken on patients
    too unwell to stand, so AP correlates with severity and therefore label.
    """

    PA = "PA"
    AP = "AP"
    LATERAL = "LATERAL"
    UNKNOWN = "UNKNOWN"


class Sex(StrEnum):
    F = "F"
    M = "M"
    UNKNOWN = "UNKNOWN"


COLUMNS: dict[str, str] = {
    "image_id": "string",
    "source": "string",
    "path": "string",
    "label": "string",
    "label_raw": "string",
    "label_provenance": "string",
    "patient_id": "string",
    "study_id": "string",
    "view": "string",
    "age": "Float64",
    "sex": "string",
    "sha256": "string",
    "phash": "string",
    "width": "Int64",
    "height": "Int64",
    # Lung segmentation mask, where the source ships one. The in-lung
    # attribution metric in Phase D is only computable for rows that have it.
    "mask_path": "string",
}

REQUIRED_NON_NULL = (
    "image_id",
    "source",
    "path",
    "label",
    "label_provenance",
    "patient_id",
    "view",
    "sha256",
)


class ManifestError(ValueError):
    """Raised when a manifest violates an invariant the gates depend on."""


def empty() -> pd.DataFrame:
    """An empty manifest with the correct dtypes."""
    return pd.DataFrame({name: pd.Series(dtype=dtype) for name, dtype in COLUMNS.items()})


def from_records(records: list[dict]) -> pd.DataFrame:
    """Build a manifest from adapter output, coercing to the declared dtypes."""
    frame = pd.DataFrame(records)
    for name, dtype in COLUMNS.items():
        if name not in frame.columns:
            frame[name] = pd.Series([None] * len(frame), dtype=dtype)
        else:
            frame[name] = frame[name].astype(dtype)
    return frame[list(COLUMNS)]


def validate(frame: pd.DataFrame) -> pd.DataFrame:
    """Check every invariant the gates rely on. Returns the frame unchanged.

    Raises ManifestError listing all violations at once, because fixing an
    adapter one error per run is miserable.
    """
    problems: list[str] = []

    missing = [name for name in COLUMNS if name not in frame.columns]
    if missing:
        raise ManifestError(f"manifest is missing columns: {sorted(missing)}")

    for column in REQUIRED_NON_NULL:
        null_count = int(frame[column].isna().sum())
        if null_count:
            problems.append(f"{column}: {null_count} null values, column is required")

    duplicated = int(frame["image_id"].duplicated().sum())
    if duplicated:
        problems.append(f"image_id: {duplicated} duplicate values, must be unique")

    problems.extend(_enum_problems(frame, "label", Label))
    problems.extend(_enum_problems(frame, "label_provenance", LabelProvenance))
    problems.extend(_enum_problems(frame, "view", View))
    problems.extend(_enum_problems(frame, "sex", Sex, allow_null=True))

    # patient_id must carry its source as a prefix. Without this, patient "1"
    # in RSNA and patient "1" in ChestX-ray14 are the same group, and G1 either
    # over-restricts the split or, if an adapter renumbers, misses real leakage.
    unprefixed = frame[
        frame["patient_id"].notna()
        & ~frame.apply(lambda row: str(row["patient_id"]).startswith(f"{row['source']}:"), axis=1)
    ]
    if len(unprefixed):
        examples = unprefixed["patient_id"].head(3).tolist()
        problems.append(
            f"patient_id: {len(unprefixed)} values not prefixed with their source "
            f"(e.g. {examples}); patient ids must be globally unique"
        )

    bad_age = frame[frame["age"].notna() & ((frame["age"] < 0) | (frame["age"] > 120))]
    if len(bad_age):
        problems.append(f"age: {len(bad_age)} values outside 0-120")

    if problems:
        raise ManifestError("manifest failed validation:\n  - " + "\n  - ".join(problems))
    return frame


def _enum_problems(
    frame: pd.DataFrame, column: str, enum: type[StrEnum], *, allow_null: bool = False
) -> list[str]:
    allowed = {member.value for member in enum}
    values = frame[column]
    present = values.dropna() if allow_null else values
    unexpected = sorted(set(present.astype(str)) - allowed)
    if unexpected:
        return [f"{column}: unexpected values {unexpected}, allowed {sorted(allowed)}"]
    return []


def write(frame: pd.DataFrame, path: Path) -> None:
    """Persist a validated manifest as Parquet."""
    validate(frame)
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(path, index=False)


def read(path: Path) -> pd.DataFrame:
    """Load and re-validate a manifest."""
    return validate(pd.read_parquet(path))


def summarise(frame: pd.DataFrame) -> pd.DataFrame:
    """Class counts per source — the table to look at before anything else.

    If a class appears under exactly one source, the model can reach that class
    by recognising the source, and no amount of later tuning will fix it.
    """
    counts = (
        frame.pivot_table(index="source", columns="label", values="image_id", aggfunc="count")
        .fillna(0)
        .astype(int)
    )
    for label in Label:
        if label.value not in counts.columns:
            counts[label.value] = 0
    counts = counts[[label.value for label in Label]]
    counts["total"] = counts.sum(axis=1)
    return counts
