# Backend — FastAPI

Two things over HTTP: store an uploaded radiograph, and score it with every
model on disk. No database, no sessions, no user accounts yet.

```sh
pip install -r ../../setup/requirements-dev.txt
MODEL_DIR=../../ml/models/serving uvicorn api.main:app --reload --port 3080
```

| Route | Does |
| --- | --- |
| `GET /health` | liveness, and whether any model actually loaded |
| `GET /models` | what is loaded and what each one is worth, without an image |
|  | — the UI's report view is built entirely from this |
| `POST /files/` | validate and store an upload, content-addressed |
| `POST /predict/` | score with every loaded model, with the evidence attached |

## `/predict` returns both models, on purpose

The result this project produced is a gap, not a score: **0.9925** macro AUC on
a corpus where COVID came from one collection, **0.7460** within a single
hospital network. A response that picked one of those would be quoting a number
without the thing that gives it meaning.

So every prediction carries its own class set, its own gate report and its own
ablation retention, and the response says what the pair is for:

> These are different questions over different corpora, not a second opinion:
> agreement between them is not corroboration, and disagreement is not a tie to
> break.

**The caveats are derived, never written out per track.** They come from the
metadata sealed inside the export, so retraining a model with different
properties cannot leave it quoting the old ones. Three fire on every response
worth naming here:

- **Retention.** Track 1 keeps 91.5% of its discriminative signal with the lung
  fields blanked out; Track 2 keeps 83.7%. The response says so in words, not
  only as a float, so a client rendering the probability still walks past it.
- **Skipped gates.** Track 2's G4 skipped because a single source makes
  class–source association undefined. A skipped gate is not a passed gate, and
  the response says which it was.
- **Out of distribution.** The quoted AUC describes held-out images from the
  collections the model trained on. An upload is from somewhere else by
  definition, so those figures do not describe the accuracy of that prediction.

## Preprocessing is imported, not reimplemented

`api.inference` imports `cxr.preprocessing.reference` — numpy and PIL only, no
torch, timm, monai or pandas, so it costs the image nothing. The model itself
comes from a `torch.export` archive, so the architecture travels with the
weights and no modelling code is needed to rebuild it.

The alternative, reimplementing the spec from `preprocessing.json`, is how
train/serve skew starts. The research package already retired its MONAI
executor over a resize that diverged from the reference by 229 times the
quantisation bound; a third implementation in the request path would be the
same mistake with a worse blast radius. `test_preprocessing_matches_the_training_reference`
pins it.

## `/models` owns its own shape

The gate object is normalised before it goes out: `blocking`, `acknowledged`,
`skipped` and `findings` are always present, whatever the stored metadata
contains. Exports made before a field existed simply omit it — Track 1 predates
`findings` — and a client following the documented shape crashed on the older
file. Filling the gaps here rather than in every consumer is the endpoint
keeping a contract it published.

## Getting models onto disk

`MODEL_DIR` holds `*.pt2` files written by `cxr export`. Each carries its
metrics, gate report and ablation retention inside the archive.

```sh
cd ../../ml
cxr export models/track2 --out models/serving/track2.pt2 \
    --ablation models/track2/ablation.json
```

An empty or missing `MODEL_DIR` is not a startup failure: the service runs,
`/health` reports `inference_available: false`, and `/predict` answers 503. A
single corrupt export is skipped and logged rather than taking the others down.

## No Django, and no accounts

The Django project is gone: `files` was ported to `api/storage.py`, and
`users`, `settings.py` and `manage.py` were deleted outright rather than
ported. There is no database and no ORM.

The upload path kept its security properties through the port, because they
were the point of that code. The client-supplied filename still never reaches
the filesystem — names are the SHA-256 of the bytes plus an extension from a
fixed map, so a request cannot choose where its file lands or smuggle a
non-image through by naming it `.png`. Those tests came across with it.

Accounts were not rebuilt. Nothing here needs to know who is asking: an upload
is written under the hash of its own bytes, scored, and forgotten. Adding
accounts would have meant a password store and a personal-data surface in
exchange for nothing this project does. The service holds no state between
requests, so there is no signing key, no session, and no record of who
uploaded what.
