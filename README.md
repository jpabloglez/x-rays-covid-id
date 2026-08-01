# X-rays COVID / non-COVID classifier

A portfolio project building a chest-radiograph classifier with an honest
evaluation story. Two models are trained. **Neither is a diagnostic device, and
demonstrating why is the point of the project.**

Train on the pooled public collections and you get **0.9925 macro AUC** with a
COVID AUC of exactly 1.0000. Blank out the lung fields entirely and **91.5% of
that signal survives** — the model is reading which collection the image came
from. Retrain within a single hospital network, where positives and negatives
share scanners and period, and the score falls to **0.7460** with 83.7% still
surviving the same ablation.

The gap between those numbers is the deliverable.
**[`ml/RESULTS.md`](ml/RESULTS.md) is the write-up.**

## Status

| Area | State |
| --- | --- |
| Image upload and display | Working |
| Dataset assembly, leakage gates | Working — five gates, measured values not pass/fail |
| Training, calibration, lung ablation | Working — two tracks trained and ablated |
| Inference API | Working — FastAPI, `POST /predict/` scores with both models |
| User accounts and organisations | Models only; the API was removed (see below) |
| Test suite | 264 over the ML package, 16 over the API |

What the app does today: you upload a JPEG or PNG, the backend validates and
stores it under a content-addressed name, and `POST /predict/` scores it with
both models — returning each one's probabilities alongside its gate report,
its lung-ablation retention and the caveats that follow from them.

**The frontend does not call `/predict/` yet.** It uploads and renders the
image back, and nothing in the interface shows a prediction; wiring that up is
Phase E. This table is the source of truth, not the UI.

The roadmap — data acquisition, leakage controls, the model ladder, calibration
and abstention, attribution overlays, reporting — is tracked separately from
this file.

## Disclaimer

This application is for demonstration purposes only and must not be used as a
substitute for professional medical diagnosis or advice. Always consult
qualified healthcare professionals for diagnosis and treatment.

Published work has repeatedly found that COVID classifiers trained on the
available public chest X-ray collections learn dataset provenance rather than
pathology — see DeGrave, Janizek & Lee (2021) and the systematic review by
Roberts et al. (2021), which found none of 415 candidate models clinically
usable. That failure mode was this project's primary hypothesis rather than an
afterthought, and **it was confirmed.** Track 1 retains 91.5% of its
discriminative signal with the lungs removed; Track 2, trained on a
single-source corpus specifically to escape the problem, still retains 83.7%.

Neither model should be quoted as a diagnostic result. They are research
artefacts that measure a documented failure mode, and their limitations are
recorded in [`ml/RESULTS.md`](ml/RESULTS.md) rather than left implicit.

The developers and contributors are not responsible for misuse or
misinterpretation of any results this application produces.

## Stack

- Backend: FastAPI + uvicorn, no database
- Frontend: React 18 + TypeScript + Vite 4 + Tailwind
- Orchestration: Docker Compose

A migration of the backend to FastAPI is planned; the Postgres service in
`docker-compose.yml` is staged for it and is not yet used.

## Prerequisites

- Docker with Compose v2

## Getting started

```sh
git clone https://github.com/jpabloglez/x-rays-covid-id.git
cd x-rays-covid-id
cp .env.example .env
docker compose up -d --build
```

There is nothing to configure before first run: no database, no migrations, no
administrator account. Mount `ml/models/serving` and the API has models; leave
it empty and it starts anyway, reporting that it has none.

- Frontend: http://localhost:3000
- Backend: http://localhost:3080
- API docs: http://localhost:3080/docs

The frontend proxies `/api` and `/media` to the backend, so the browser stays
same-origin and no CORS configuration is needed for local development.

## Running without Docker

```sh
python -m venv .venv && source .venv/bin/activate
pip install -r setup/requirements-dev.txt
cd app/backend
MODEL_DIR=../../ml/models/serving uvicorn api.main:app --reload --port 3080
```

```sh
cd app/frontend && npm ci && npm run dev
```

## Configuration

Every environment-dependent value is read from the environment; `.env.example`
documents the full list, and none of it is secret. The API keeps no accounts,
no sessions and no database, so there is no signing key to protect and no
credentials to rotate.

## Tests and linting

```sh
pip install -r setup/requirements-dev.txt
cd app/backend && pytest
ruff check .
```

## Layout

```
app/backend/    FastAPI service: upload, and inference over both tracks
app/frontend/   React + Vite single-page app
compose/        Dockerfiles for the backend and frontend images
setup/          Python requirements
```

## No accounts, by decision

The original repository carried a user system: accounts with a five-level role
field, organisations with billing addresses, profiles with phone numbers and
avatars. Its API exposed list, retrieve, update and delete over every account
with no authentication at all, behind permission classes that raised
`UnboundLocalError` before they could deny anything. Phase A deleted the views;
this branch deleted the rest.

It was not replaced. Nothing this project does needs to know who is asking —
you upload a radiograph, it is scored, and the answer comes back with its
confound profile attached. Accounts would have added a password store, a
personal-data surface and a login wall in front of the one thing worth looking
at, in exchange for nothing.

So there is no database, no session, no signing key, and no record of who
uploaded what. An application that holds no personal data cannot leak personal
data, which is a stronger guarantee than any amount of careful handling.

## License

MIT — see `LICENSE`.
