# Convenience wrapper around the commands used to run and check this project.
# Nothing here is required -- every target is a documented shorthand for a
# command already described in one of the READMEs.
#
# Destructive scope, on purpose: this Docker daemon also runs containers and
# images for other, unrelated projects on this machine. `clean` only ever
# touches what this docker-compose.yml itself created (`--rmi local` removes
# locally built images, not pulled ones; nothing here calls a bare
# `docker system prune`, which would take other projects' images with it).

.PHONY: help up up-build down restart build ps logs logs-backend logs-frontend \
        backend-shell frontend-shell export-models test test-ml test-backend \
        test-frontend lint clean

help: ## Show this list
	@grep -E '^[a-zA-Z0-9_-]+:.*## ' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*## "}; {printf "  \033[36m%-16s\033[0m %s\n", $$1, $$2}'

up: ## Start the stack in the background, using whatever images already exist
	docker compose up -d

up-build: ## Start the stack, rebuilding images first
	docker compose up -d --build

down: ## Stop the stack, keep images and volumes
	docker compose down

restart: down up ## down, then up

build: ## Build both images without starting anything
	docker compose build

ps: ## Show what's running
	docker compose ps

logs: ## Follow logs from every service
	docker compose logs -f

logs-backend: ## Follow only the backend's logs
	docker compose logs -f backend-xrays

logs-frontend: ## Follow only the frontend's logs
	docker compose logs -f frontend-xrays

backend-shell: ## Open a shell inside the running backend container
	docker compose exec backend-xrays bash

frontend-shell: ## Open a shell inside the running frontend container
	docker compose exec frontend-xrays bash

export-models: ## Freeze both trained checkpoints for the API to serve
	cd ml && ./.venv/bin/python -m cxr.cli export models/track1 --out models/serving/track1.pt2 \
	    --ablation reports/ablation-full.json
	cd ml && ./.venv/bin/python -m cxr.cli export models/track2 --out models/serving/track2.pt2 \
	    --ablation models/track2/ablation.json --gates ../datasets/manifests/bimcv-gates.json

test: test-ml test-backend test-frontend ## Run all three test suites, as CI does

test-ml: ## Run the ML package's tests
	cd ml && ./.venv/bin/pytest

test-backend: ## Run the API's tests
	cd app/backend && PATH="../../ml/.venv/bin:$$PATH" pytest

test-frontend: ## Run the frontend's tests
	cd app/frontend && npx vitest run

lint: ## Ruff, over the whole repo
	ml/.venv/bin/python -m ruff check .

clean: ## Stop the stack and remove ITS OWN containers, local images and volumes
	docker compose down --rmi local -v --remove-orphans
