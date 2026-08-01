# `cxr` — dataset assembly and leakage gates

Research code for the chest radiograph classifier: turning downloaded datasets
into a validated manifest, splitting it so the splits mean something, refusing
to proceed when the corpus is confounded, and then measuring what a score is
actually made of.

Two models are trained. Track 1 scores 0.9925 macro AUC and keeps 91.5% of that
signal with the lung fields blanked out. Track 2 scores 0.7460 and keeps 83.7%.
**[RESULTS.md](RESULTS.md) is the write-up**, and the gap between those two
numbers is the finding — neither is a diagnostic device.

This package never imports the web application, and the application never
imports it. They will communicate through exactly two artifacts: a serialised
model and a preprocessing config.

## Why the gates exist

Published COVID chest X-ray classifiers routinely report 98–99% accuracy.
DeGrave, Janizek & Lee (2021) showed those numbers largely come from shortcut
learning, and Roberts *et al.* (2021) reviewed 415 papers and found none
clinically usable. The dominant cause is **source confounding**: positives from
one repository, negatives from another, so the network learns the scanner, the
crop convention or the burnt-in marker rather than the pathology.

For Track 1 that is not an avoidable mistake — it is forced by the available
data. ChestX-ray14 (2017), RSNA Pneumonia (2018) and CheXpert (2019) all
predate the pandemic and carry no COVID label, so a three-class corpus needs a
fourth, pandemic-era source, and at that moment class becomes predictable from
provenance. The gates measure how badly rather than letting it pass unnoticed:
G3 found source 79.5% predictable from 32×32 thumbnails, and G4 failed by
construction.

Registering BIMCV-COVID19 is what makes the second track possible. It publishes
molecularly positive *and* negative patients from one hospital network, so
class and collection are decoupled without a second source having to be
trusted. `cxr sources` now lists two collections that can supply a COVID label,
and the warning that once said only one could no longer fires — that line was
driven by the registry, not hardcoded, so it retired itself.

## The gates

| Gate | Asserts | Catches |
| --- | --- | --- |
| **G1** | No `patient_id` spans two splits | A follow-up film of the same chest in both train and test |
| **G1b** | The external split is exactly one unseen source | An "external" set that is really an internal one |
| **G2** | No perceptual-hash cluster spans two splits | Aggregate collections reposting each other's images, re-encoded |
| **G3** | Source is not predictable from pixels | Acquisition signatures a network can read for free |
| **G4** | Class and source are independent | How much of the label is available from provenance alone |

Each returns a measured value, not just pass/fail — `G4 passed` is far less
useful than `Cramér's V 0.31, threshold 0.40`, and the measured numbers are
what the model card publishes.

## Usage

```sh
pip install -e ml                 # add [imaging] for DICOM sources
cxr sources                       # what is available and what it lacks
cxr assemble --source chestxray14 --root data/raw/nih --out data/manifests/nih.parquet
cxr assemble --source covid_radiography --root data/raw/covid --out data/manifests/covid.parquet
cxr merge data/manifests/*.parquet --out data/manifests/corpus.parquet
cxr dedupe data/manifests/corpus.parquet --out data/manifests/corpus-dedup.parquet --drop
cxr split data/manifests/corpus-dedup.parquet --out data/manifests/split.parquet --holdout-source covid_radiography
cxr gates data/manifests/split.parquet --images data/raw --json reports/gates.json
```

`cxr gates` exits non-zero when a gate fails. It is meant to sit in CI between
assembling data and training on it.

**Leakage and confounds are handled differently.** G1, G1b and G2 are defects
and always block: an image on both sides of a split makes every number measured
afterwards meaningless. G3 and G4 are findings, and Track 1 trains on a
confounded corpus deliberately, so they can be waived by name:

```sh
cxr gates data/manifests/split.parquet --images ... --acknowledge G3,G4
```

Naming a gate that then passes is an error rather than a no-op, so the list
cannot go stale and quietly disarm a live check. What was acknowledged is
written into `gates.json` for the model card, because reporting a confounded
corpus without recording that the confound was known in advance describes
different work.

## Training

```sh
cxr cache data/manifests/split.parquet --out data/cache/track1-320 \
    --images covid_radiography=... --images rsna_pneumonia=... --target-size 320
cxr train data/manifests/split.parquet --out models/track1 \
    --cache data/cache/track1-320 --gates reports/gates.json
```

`train` reads the gate report and refuses a split with blocking failures, so
the two are meant to run in that order. Acknowledged confounds are carried into
the checkpoint and printed above every score.

The class set is a property of the task, not of the module. `--task` selects it
explicitly; omitted, it is derived by matching the manifest's labels against the
registered tasks, so the ordering stays canonical and labels matching no task
are refused rather than guessed. `ModelConfig.classes` is the single
declaration — taking it from anywhere else would let the labels and the output
layer describe different problems, and the run would train quite happily on the
mismatch.

**A score is not a result until the ablation has run.** `cxr ablate` blanks the
lung fields and scores again, in both directions, because either alone is
ambiguous. Track 1 keeps 91.5% of its signal with the anatomy removed; that
number belongs beside the 0.9925, not in a footnote. See
[RESULTS.md](RESULTS.md).

**Install torch for your GPU before the extras.** Which CUDA build you get
matters more than which release. A wheel compiled for newer architectures than
your card installs cleanly, reports `torch.cuda.is_available()` as `True`, and
then fails at the first kernel launch with `no kernel image is available for
execution on the device`. That happened here: `timm` and `monai` pulled
`torch 2.13+cu130` as a transitive dependency onto a GTX 1050, whose `sm_61` is
below that build's `sm_75` floor. `cu126` works, because it ships `sm_60` and
CUDA cubins are forward-compatible across minor revisions.

```sh
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126
```

`resolve_device()` probes with a real matmul rather than trusting the
availability flag, so a mismatch falls back to CPU with the card, the build's
arch list and the fix printed — instead of crashing after the model is built.

**Preprocessing is cached, augmentation is not.** Decoding 14,863 DICOMs
through the VOI LUT every epoch would make data loading, not the GPU, decide
how long a run takes. Caching the augmentation too would mean every epoch saw
the same "random" transform, which is the same as not augmenting.

**The cache is uint8, and that was measured rather than assumed.** It is a
train/serve difference, which is what retired the two-executor design — so the
round trip was measured on 80 real radiographs: 0.00875 in normalised units,
exactly the round-to-nearest bound. The augmentation the model is trained to
tolerate is ten times larger; the resize divergence that killed the
two-executor design was 229 times larger and exceeded the normalisation scale.
`test_cache.py` pins the bound.

## Design decisions worth knowing

**Patient ids are namespaced by source.** Patient `1` in RSNA and patient `1`
in ChestX-ray14 are different people. Validation rejects any unprefixed id,
because a collision makes G1 report a disjointness it never verified.

**Sources that lack patient ids say so.** The COVID-19 Radiography Database
publishes none, so its adapter synthesises one per image and G1 is blind for
those rows. That is recorded, not papered over.

**"Other pathology" is not "normal".** ChestX-ray14 rows with a finding that
is not pneumonia are excluded rather than relabelled normal; RSNA's
`No Lung Opacity / Not Normal` likewise. Relabelling them teaches the model
that abnormal chests are healthy, and it is the most common misuse of both.

**Unknown means unknown.** The COVID source does not record projection, so its
`view` is `UNKNOWN` rather than assumed `PA` — an assumption there would
corrupt the AP/PA stratification that the bias analysis depends on.

**The near-duplicate threshold needs calibrating on real data.** Chest
radiographs share their gross anatomy, so their hashes sit closer together than
natural images and a threshold tuned on photographs will merge unrelated
patients. Cluster at several thresholds; if a cluster contains two different
`patient_id`s that are not a known duplicate pair, it is too loose. On the
first real corpus a 64-bit hash was unusable — the nearest *distinct* pair sat
one bit away — and widening to 256 bits gave a clean 23-bit gap.

**The collections overlap far more than their documentation suggests.** 8,850
of the RSNA Pneumonia Challenge's 8,851 normal studies have a twin in the
COVID-19 Radiography Database's normal class: the latter took its normal class
from the former. Pooling the two without deduplicating puts 59% of the corpus
on both sides of a split. `cxr dedupe --drop` keeps the RSNA DICOM, which is
both the higher-fidelity original and the choice that leaves the non-COVID
classes mixed across sources rather than aligned with them.

**Duplication also suppresses G3.** A source probe cannot beat chance on two
pixel-identical images labelled with different sources, so a corpus that is 59%
twins caps the probe near 0.70 whatever the acquisition differences are. Read
G3 only after G2 is clean; before that, a low reading measures duplication, not
the absence of a confound.

**G4 skips splits too small to score.** Below an expected cell count of five
the chi-square approximation breaks down, and a small calibration slice would
otherwise report a large association that is pure sampling noise. A gate that
cries wolf gets switched off.

## Tests

```sh
cd ml && pytest
```

Every gate is tested twice: that it passes clean data, and that it *fires* on
deliberately planted leakage. The second half is the one that matters — a
leakage detector only ever run on data believed clean would pass identically if
it were a function returning `True`.

Fixtures are synthetic, which is the only way to test a leakage detector
properly: the test needs to plant the leakage and assert it is found. On real
data you never know the ground truth of how confounded your corpus is, which is
the situation these gates exist to escape.
