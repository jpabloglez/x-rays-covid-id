# Results — two tracks, and the distance between them

Two models, trained by the same code on the same day, differing only in what
corpus they were given. Track 1 scores **0.9925**. Track 2 scores **0.7460**.
Neither is a COVID detector, and the reason is the point of this document.

The sharpest single result is at the bottom of the confounds section: **a
lookup table that reads only which x-ray machine took the image, and never
looks at the image, beats the trained network on Track 2's own test split.**

All numbers below are measured, on held-out test splits, with the commands that
produced them recorded. Nothing here is an estimate.

---

## The two tracks

| | Track 1 | Track 2 |
| --- | --- | --- |
| Question | What can a pooled corpus be made to score? | Is COVID separable from other reasons to be x-rayed, within one hospital network? |
| Classes | `normal`, `pneumonia`, `covid` | `non_covid`, `covid` |
| Sources | RSNA Pneumonia + COVID-19 Radiography | BIMCV-COVID19 only |
| Images | 30,016 | 1,463 |
| Patients | 30,016 | 738 |
| Class ↔ source | **confounded by construction** | decoupled by construction |

Track 1 is confounded on purpose. Every pre-pandemic collection — ChestX-ray14,
RSNA, CheXpert — predates COVID and carries no COVID label, so a three-class
corpus needs a pandemic-era source, and at that moment provenance predicts the
class. Track 1 measures how much that is worth. Track 2 exists to answer the
same question with provenance removed: BIMCV publishes molecularly positive and
negative patients from the same Valencian hospital network, the same scanners,
the same period.

---

## Headline metrics

| | Track 1 | Track 2 |
| --- | --- | --- |
| Test macro AUC | **0.9925** | **0.7460** |
| Balanced accuracy | 0.9653 | 0.6479 |
| COVID AUC | **1.0000** | 0.7460 |
| ECE before calibration | 0.0298 | 0.1814 |
| ECE after calibration | 0.0069 | 0.0854 |
| Temperature | 2.24 | 3.35 |
| Test images | 4,168 | 293 |

A COVID AUC of **exactly 1.0000** should be read as a measurement of the
corpus, not of the model. Every COVID image in Track 1 came from one
collection; perfect separation is what that looks like from the inside.

Track 2's calibration is poor and its corpus is small. Training loss fell
0.71 → 0.12 while validation AUC sat flat near 0.68 — plain overfitting on 790
training images. A temperature of 3.35 is the correction for it. Quote 0.7460
with that caveat attached.

---

## The leakage gates

| Gate | Track 1 | Track 2 |
| --- | --- | --- |
| G1 patient disjointness | pass | pass |
| G1b external split purity | skipped | skipped |
| G2 near-duplicate disjointness | pass (after dedupe) | pass |
| G3 source-confound probe | **fail — 0.795**, acknowledged | skipped |
| G4 class–source independence | **fail**, acknowledged | skipped |
| G5 scanner–class independence | not measured | **fail — 0.474**, acknowledged |

**G3 = 0.795** means a probe distinguishes the two collections from 32×32
thumbnails four times in five. That signal is free to any network trained on
the pooled corpus.

**Track 2's G4 skips rather than passes, and the distinction matters.** With
one source, class–source association is undefined — there is nothing to
measure. Track 2's freedom from source confounding is true *by construction*,
not by test. Reporting it as a pass would claim a measurement that never
happened.

G2 is worth recording as a near-miss: dedupe found **8,850 cross-source
duplicate pairs** between RSNA and COVID-19 Radiography — the same radiographs
reposted at different resolutions, zero label conflicts among them. Left in,
they would have put the same chest on both sides of the split.

---

## The lung ablation — what the scores are made of

Blank the lung fields and score again. A model reading pathology should
collapse towards chance; one reading acquisition signature barely notices.
Retention is the ablated AUC rescaled between chance and the unablated score on
**the same rows**.

Both directions are run, because either alone is ambiguous.

| | Track 1 | Track 2 |
| --- | --- | --- |
| Baseline on covered rows | 0.9925 | 0.7473 |
| **Lungs removed** | 0.9507 → **91.5%** retained | 0.7070 → **83.7%** retained |
| **Lungs only** | 0.9394 → 89.2% retained | 0.6151 → 46.5% retained |
| Covered images | 4,168 | 292 |

### What this says

**Track 1 barely notices losing the lungs.** 91.5% of its above-chance signal
survives having the anatomy blanked out. It is reading the collection.

**Track 2 is better and still not good.** 83.7% survives. Decoupling class from
source cost 0.25 AUC and bought roughly eight points of shortcut reduction.
Keeping *only* the lungs retains 46.5% — so the model cannot do the job from
anatomy alone even within a single hospital.

The two retentions sum to more than 100% in both tracks, which means the signal
is partly redundant rather than cleanly partitioned: some of it is available
both inside and outside the lung fields. That is consistent with a confound
correlated with the anatomy without being it.

**The honest statement for Track 2 is not "0.746". It is "0.746, of which most
is not anatomy."**

---

## Confounds measured but not removed

Track 2 escapes the source confound. It does not escape these.

| Confound | Cramér's V | Direction |
| --- | --- | --- |
| Label ↔ view position | 0.152 | COVID 57% AP vs non-COVID 41% AP |
| Label ↔ patient sex (per patient) | 0.100 | COVID 57% male vs non-COVID 46% |

Both are real epidemiology and both are visible in the image. The sickest
patients get a portable AP film; COVID hospitalisation skewed male. A model can
reach 0.746 partly by learning "portable AP" or "male chest" — neither of which
is the disease.

Measured at image level the sex association reads 0.198, but that is inflated
because COVID patients receive more follow-up films. Per patient is the honest
figure.

### The scanner, which turned out to be the answer

Within one collection, source is constant and the acquisition signature that
remains is the individual device. Measuring it settled what Track 2's 83.7%
extra-pulmonary signal was.

**Cramér's V = 0.474** between scanner model and class, over 42 devices —
above the 0.40 threshold, so G5 fails and blocks. Bias-corrected, because with
that many thin levels an uncorrected V rises with the level count whether or
not any association exists.

COVID rate by device, corpus-wide, for devices with 40 or more films:

| device | films | covid |
| --- | ---: | ---: |
| CR 85 | 60 | 85% |
| CS-7 | 68 | 76% |
| 0862 | 48 | 73% |
| DX-G | 60 | 67% |
| ACCORD DR | 42 | 55% |
| DR 14e C - 1200ms | 120 | 53% |
| DRX-1 | 94 | 46% |
| Varian_4343R | 44 | 43% |
| DigitalDiagnost | 68 | 35% |
| SIEMENS FD-X | 81 | 33% |
| DX-M | 51 | 29% |
| *unrecorded* | 219 | 20% |
| DRX-Evolution | 79 | 19% |
| PCR Eleva | 43 | 19% |
| 3543EZE | 43 | 12% |

Ordinary hospital logistics rather than exotic leakage: a portable unit is
wheeled to the COVID ward, the radiology suite takes scheduled outpatients.
Different detectors, different processing, different noise. BIMCV's
single-source purity does nothing about it.

**A lookup table beats the model.** Estimate P(covid | device) on `train`,
apply it to `test`, and compare against the network scored on the same 293
images:

| | AUC | balanced accuracy |
| --- | --- | --- |
| **Scanner lookup**, never sees the image | **0.7640** | **0.7522** |
| **DenseNet-121**, 790 training images | 0.7460 | 0.6479 |

Only 4 of 293 test rows sit on a device unseen in training, so this is not an
artefact of the fallback.

**What that means for Track 2's number.** 0.7460 should not be read as "COVID
detection at 0.75". Track 2 was built to escape source confounding and it
succeeded — G4 genuinely cannot associate class with collection. It then
walked into the confound one level down, and measured the scanner less
efficiently than a lookup table would.

Rows with no device recorded are kept as their own level rather than dropped.
Which images lack metadata is itself an acquisition signature, and 14.4% of
the corpus is a large thing to delete on the way to a reassuring number.

---

## Segmentation, and why its Dice is not evidence about the target

BIMCV ships no lung masks, so they were predicted. A MONAI U-Net was fitted on
COVID-19 Radiography's 3,249 masked training images and applied to BIMCV.

- **Dice 0.9729** on 1,216 held-out images — *of the source collection only*.
- **1,455 of 1,463** BIMCV masks passed the shape-plausibility gate (99.5%).

There is no ground truth on BIMCV, so no Dice can be quoted there. The masks
are judged on shape alone — area fraction, left/right imbalance, vertical
centroid, connected components — which rejects the two ways domain transfer
actually fails: predicting nothing, and predicting everything.

The evidence that transfer held is the distribution comparison, not the Dice:

| | area | imbalance | centroid |
| --- | --- | --- | --- |
| covid_radiography (source) | 0.233 | 0.127 | 0.446 |
| bimcv_covid19 (target) | 0.224 | 0.143 | 0.459 |

Near-identical. A segmenter failing on the target would show up here first.

---

## Limits of this report

- **Track 2's test set is 293 images from 148 patients.** Confidence intervals
  are wide and not yet computed.
- **The ablation covers one collection.** It cannot separate anatomy from
  BIMCV's own acquisition signature; only a second single-source corpus could.
- **G3 never ran for Track 2**, so the within-source pixel confound is
  unquantified. G5 measures the recorded device, which is not the same thing:
  a probe on pixels could find acquisition signature that no metadata field
  names.
- **14.4% of BIMCV rows record no device.** They are kept as their own level,
  which is the conservative choice, but a large unlabelled group limits how
  precisely the scanner association can be pinned.
- **Neither model has an external validation set.** G1b skipped in both.
- **`label` is molecular status, not the radiologist's reading.** The two
  disagree on roughly a fifth of BIMCV: 13.8% of molecularly positive studies
  read as radiologically normal. A model asked to predict PCR status from an
  image is partly being asked to predict something the image does not contain.

---

## Neither model is a diagnostic device

Track 1 reports 0.9925 and reads the collection. Track 2 reports 0.7460 and is
beaten, on its own test split, by a lookup table that reads only which machine
took the image. Both are research artefacts produced to quantify a documented
failure mode, and the gap between them is the result — not either number on
its own.

Track 2 is the more interesting failure. It was designed to remove the
confound Track 1 has, it removed exactly that confound, and it was defeated by
the next one down. Escaping one shortcut is not the same as reading the
anatomy.

DeGrave, Janizek & Lee (2021) showed published COVID classifiers doing exactly
what Track 1 does here. Roberts *et al.* (2021) reviewed 415 papers and found
none clinically usable. This repository reproduces that finding deliberately
and measures it, which is the only claim it makes.

---

## Reproducing

```sh
# Track 2, end to end
cxr assemble --source bimcv_covid19 --root datasets/bimcv --out m/bimcv.parquet
cxr dedupe   m/bimcv.parquet --out m/bimcv-dedup.parquet
cxr split    m/bimcv-dedup.parquet --out m/bimcv-split.parquet
cxr gates    m/bimcv-split.parquet --json m/bimcv-gates.json
cxr cache    m/bimcv-split.parquet --out cache/bimcv \
             --images bimcv_covid19=datasets/bimcv --target-size 320
cxr train    m/bimcv-split.parquet --out models/track2 \
             --cache cache/bimcv --gates m/bimcv-gates.json --epochs 30

# masks, then the ablation that decides what the score is worth
cxr segment  m/segment-bimcv.parquet --out datasets/masks/bimcv \
             --masks covid_radiography=datasets/COVID-19_Radiography_Dataset \
             --cache cache/segment-bimcv --manifest-out m/bimcv-masked.parquet
cxr ablate   m/bimcv-ablate.parquet --checkpoint models/track2 \
             --masks bimcv_covid19=datasets/masks/bimcv \
             --cache cache/bimcv --split test --json models/track2/ablation.json
```

Seeds are fixed at 0 throughout and recorded in every checkpoint alongside the
gate report, the class set and the sources the run actually saw.
