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
printf 'POSTGRES_PASSWORD=%s\n' "$(openssl rand -hex 24)" >> .env
docker compose up -d --build
```

No credential in this repository has a default value. `POSTGRES_PASSWORD` is
unset in `.env.example` on purpose, and compose refuses to start the database
service until you generate one.

- Frontend: http://localhost:3000
- Backend: http://localhost:3080
- Django admin: http://localhost:3080/admin/

Apply migrations and create an administrator on first run:

```sh
docker compose exec backend-xrays python manage.py migrate
docker compose exec backend-xrays python manage.py createsuperuser
```

The frontend proxies `/api` and `/media` to the backend, so the browser stays
same-origin and no CORS configuration is needed for local development.

## Running without Docker

```sh
python -m venv .venv && source .venv/bin/activate
pip install -r setup/requirements-dev.txt
cd app/backend
DJANGO_DEBUG=true python manage.py migrate
DJANGO_DEBUG=true python manage.py runserver 0.0.0.0:3080
```

```sh
cd app/frontend && npm ci && npm run dev
```

## Configuration

Every environment-dependent value is read from the environment; `.env.example`
documents the full list. `DJANGO_SECRET_KEY` is mandatory whenever
`DJANGO_DEBUG` is off — the app refuses to start without it rather than falling
back to a key committed to the repository.

## Tests and linting

```sh
pip install -r setup/requirements-dev.txt
cd app/backend && pytest
ruff check .
```

## Layout

```
app/backend/    Django project: files (upload) and users (models only)
app/frontend/   React + Vite single-page app
compose/        Dockerfiles for the backend and frontend images
setup/          Python requirements
```

## Removed user API

`users/` previously exposed list, retrieve, update and delete endpoints for
every account with no authentication, backed by permission classes that raised
`UnboundLocalError` before they could deny anything. Those views, URLs,
serializers and permissions have been deleted. The models remain because
`AUTH_USER_MODEL` points at them; the API will be rebuilt with real
authentication during the FastAPI migration.

## Security note

An earlier revision of this repository committed `app/backend/db.sqlite3`,
including two user rows with password hashes, and a hardcoded `SECRET_KEY`.
Both are removed from the working tree. Until the history is rewritten those
values remain reachable in earlier commits and must be treated as compromised.

## License

MIT — see `LICENSE`.
