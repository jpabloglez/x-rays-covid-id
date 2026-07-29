"""BIMCV-COVID19 iterations 1+2, positive and negative partitions.

The source Track 2 needs, and the only one here where COVID status is not a
proxy for provenance: both classes come from the same Valencian hospital
network, the same scanners and the same period, so recognising the collection
tells a model nothing about the label. That is the whole point of it.

Three things about this data need handling rather than noting.

**The label has two independent definitions.** The partition -- `covid19_posi`
or `covid19_neg` -- is molecular status. The `Labels` column in
`derivatives/labels/` is what the radiologist wrote. They disagree on roughly a
fifth of the corpus: 13.8% of molecularly positive studies are radiologically
normal, and 6.1% of the negative partition mentions COVID. Neither is wrong;
they answer different questions. `label` carries molecular status, because that
is the ground truth Track 2 is about, and the radiological reading is not
silently merged into it.

Note what `label_raw` does *not* contain. The per-image session TSVs carry a
DICOM Series Description, which is a protocol name -- "W033 Tórax PA" -- not a
finding. The radiological readings live in `derivatives/labels/*.tsv`, keyed by
report rather than by image, and joining them needs a report-to-session mapping
this adapter does not yet do. Until it does, `label_raw` states the partition
the image came from, which is exactly what the source asserts about it.

**Negatives are not normal.** Only a fifth of the negative partition is
radiologically clear; the rest carries effusion, pneumonia, cardiomegaly. They
are labelled `non_covid`, never `normal`. Calling them normal would turn Track
2 into sick-versus-healthy and reproduce the confound it exists to escape.

**Lateral films hide from the obvious filter.** Most images encode projection
in the filename as `vp-pa`, `vp-ap` or `vp-ll`, but some carry no `vp-` marker
at all -- and in the sample measured here, every unmarked file was a lateral,
identified only by a Spanish Series Description of "Tórax Lat.". Filtering on
`vp-ll` alone therefore admits laterals as unknown-view.

That matters more than the count. A lateral is not a hard frontal case, it is a
different projection entirely, and if laterals are distributed unevenly across
COVID status -- plausible, since the sickest patients get a portable AP and no
lateral -- then "is this a lateral" becomes a label proxy. So only images
explicitly marked frontal are admitted. Unknown projection is excluded rather
than assumed, which is the same rule the rest of this package applies.
"""

from __future__ import annotations

import csv
import re
from pathlib import Path

from cxr.manifest import Label, LabelProvenance, Sex, View
from cxr.sources.base import Source, SourceError, SourceInfo

PARTITIONS = {"covid19_posi": Label.COVID, "covid19_neg": Label.NON_COVID}

# Only these reach the manifest. `ll` is lateral and excluded by name; anything
# without a marker is excluded too, because the unmarked files observed here
# were laterals whose projection lived only in a free-text description.
FRONTAL = {"pa": View.PA, "ap": View.AP}

_VIEW = re.compile(r"_vp-([a-z]+)[_.]")
_SUBJECT = re.compile(r"(sub-S\d+)")
_SESSION = re.compile(r"(ses-E\d+)")
_LATERAL_TEXT = re.compile(r"\blat\b|lateral", re.IGNORECASE)


class BimcvCovid19(Source):
    info = SourceInfo(
        name="bimcv_covid19",
        title="BIMCV-COVID19+ / COVID19- (iterations 1+2)",
        citation="Vayá et al. 2020, arXiv:2006.01174",
        licence="CC BY 4.0; non-commercial research, attribution required",
        has_covid_label=True,
        has_patient_ids=True,
        has_demographics=True,
        modality="PNG",
        notes=(
            "The only source here where COVID status is not confounded with "
            "provenance: positives and negatives share a hospital network, so "
            "this is what Track 2 is built on. Negatives are labelled "
            "non_covid, never normal -- only a fifth are radiologically clear. "
            "Lateral projections are excluded, including the ones that carry no "
            "vp- marker and are identifiable only from the series description."
        ),
    )

    def build(self, root: Path) -> list[dict]:
        root = Path(root)
        present = [name for name in PARTITIONS if (root / name).is_dir()]
        if not present:
            raise SourceError(
                f"no BIMCV partition found under {root}; expected at least one of "
                f"{sorted(PARTITIONS)}"
            )

        records: list[dict] = []
        for partition in present:
            metadata = _scan_metadata(root / partition)
            for image in sorted((root / partition).rglob("*.png")):
                record = self._record(image, root, partition, metadata)
                if record is not None:
                    records.append(record)

        if not records:
            raise SourceError(
                f"no frontal chest radiographs found under {root}. Laterals and "
                "unknown-projection images are excluded by design; check that the "
                "download extracted the mod-rx directories."
            )
        return records

    def _record(self, image: Path, root: Path, partition: str, metadata: dict) -> dict | None:
        name = image.name
        if "bp-chest" not in name:
            return None

        marker = _VIEW.search(name)
        if marker is None or marker.group(1) not in FRONTAL:
            return None

        row = metadata.get(name, {})
        # Belt and braces: a file marked frontal whose description says lateral
        # is a contradiction, and the safe reading is to drop it.
        if _LATERAL_TEXT.search(row.get("Series Description (0008103E)", "")):
            return None

        subject = _SUBJECT.search(str(image))
        session = _SESSION.search(str(image))
        if subject is None:
            return None

        relative = image.relative_to(root)
        return {
            "image_id": f"{self.info.name}:{relative.as_posix()}",
            "source": self.info.name,
            "path": str(relative),
            "label": str(PARTITIONS[partition]),
            # The partition, which is what the source actually asserts. Not the
            # Series Description: that is a protocol name ("W033 Tórax PA"),
            # and putting it here would look like a radiological reading while
            # carrying none.
            "label_raw": partition,
            "label_provenance": str(LabelProvenance.PCR),
            "patient_id": f"{self.info.name}:{subject.group(1)}",
            "study_id": session.group(1) if session else None,
            "view": str(FRONTAL[marker.group(1)]),
            "age": _map_age(row.get("Patient's Age (00101010)")),
            "sex": _map_sex(row.get("Patient's Sex (00100040)")),
            "mask_path": None,
            **self.hash_record(image, relative),
        }


def _scan_metadata(partition_root: Path) -> dict[str, dict]:
    """Per-image DICOM metadata, keyed by filename, from the session TSVs.

    Manufacturer, model and series description are kept deliberately.
    Within a single collection
    they are the finest-grained acquisition signature available, which makes
    them the right thing to run a confound probe against once Track 2 has a
    score -- source is constant here, so scanner is the confound that remains.
    """
    metadata: dict[str, dict] = {}
    for tsv in partition_root.rglob("*_scans.tsv"):
        try:
            with tsv.open(newline="", encoding="utf-8", errors="replace") as handle:
                for row in csv.DictReader(handle, delimiter="\t"):
                    filename = (row.get("filename") or "").strip()
                    if filename:
                        metadata[filename] = row
        except OSError:
            # A missing or unreadable session file costs metadata, not the
            # image; the row is still usable with unknown demographics.
            continue
    return metadata


def _map_age(value: object) -> float | None:
    text = str(value or "").strip().upper().rstrip("Y").lstrip("0")
    if not text or text == "N/A":
        return None
    try:
        age = float(text)
    except ValueError:
        return None
    return age if 0 <= age <= 120 else None


def _map_sex(value: object) -> str:
    text = str(value or "").strip().upper()
    return text if text in {Sex.F, Sex.M} else str(Sex.UNKNOWN)
