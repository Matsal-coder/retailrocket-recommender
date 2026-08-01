.DEFAULT_GOAL := help

POETRY ?= poetry
RUN := $(POETRY) run
COMPOSE := docker compose

PYTHON := $(RUN) python
PYTEST := $(RUN) pytest
RUFF := $(RUN) ruff
DVC := $(RUN) dvc

MLFLOW_HOST ?= 127.0.0.1
MLFLOW_PORT ?= 5000
MLFLOW_WORKERS ?= 1
MLFLOW_BACKEND_STORE_URI ?= sqlite:///mlflow.db
MLFLOW_ARTIFACT_ROOT ?= ./mlartifacts

.PHONY: help
help: ## Show the available project commands
	@echo ""
	@echo "RetailRocket Recommender"
	@echo "========================"
	@echo ""
	@awk 'BEGIN {FS = ":.*## "}; /^[a-zA-Z0-9_-]+:.*## / {printf "  %-26s %s\n", $$1, $$2}' $(MAKEFILE_LIST)
	@echo ""

.PHONY: install
install: ## Install main and development dependencies
	$(POETRY) install --with dev

.PHONY: lock
lock: ## Regenerate the Poetry lock file
	$(POETRY) lock

.PHONY: check-dependencies
check-dependencies: ## Validate pyproject.toml and poetry.lock
	$(POETRY) check

.PHONY: env
env: ## Validate the local project environment
	$(PYTHON) scripts/validate_env.py

.PHONY: test
test: ## Run the complete test suite
	$(PYTEST)

.PHONY: test-unit
test-unit: ## Run unit tests
	$(PYTEST) tests/unit -v

.PHONY: test-integration
test-integration: ## Run integration tests
	$(PYTEST) tests/integration -v

.PHONY: coverage
coverage: ## Run tests and generate the HTML coverage report
	$(PYTEST) --cov=retail_recommender --cov-report=term-missing --cov-report=html

.PHONY: lint
lint: ## Run Ruff lint checks
	$(RUFF) check .

.PHONY: lint-fix
lint-fix: ## Apply safe Ruff lint fixes
	$(RUFF) check . --fix

.PHONY: format
format: ## Format the project with Ruff
	$(RUFF) format .

.PHONY: format-check
format-check: ## Check formatting without changing files
	$(RUFF) format --check .

.PHONY: quality
quality: lint format-check test ## Run lint, formatting and tests

.PHONY: pre-commit
pre-commit: ## Run all pre-commit hooks
	$(RUN) pre-commit run --all-files

.PHONY: validate-data
validate-data: ## Run the data validation pipeline directly
	$(PYTHON) -m retail_recommender.pipelines.validate_data

.PHONY: preprocess
preprocess: ## Run the preprocessing pipeline directly
	$(PYTHON) -m retail_recommender.pipelines.preprocess

.PHONY: feature-engineering
feature-engineering: ## Run the feature engineering pipeline directly
	$(PYTHON) -m retail_recommender.pipelines.feature_engineering

.PHONY: train
train: ## Run the training pipeline directly
	$(PYTHON) -m retail_recommender.pipelines.train

.PHONY: evaluate
evaluate: ## Run the evaluation and model-selection pipeline directly
	$(PYTHON) -m retail_recommender.pipelines.evaluate

.PHONY: register-model
register-model: ## Register the selected model in MLflow
	$(PYTHON) -m retail_recommender.pipelines.register_model

.PHONY: dvc-dag
dvc-dag: ## Display the DVC pipeline DAG
	$(DVC) dag

.PHONY: dvc-status
dvc-status: ## Check the status of the DVC pipeline
	$(DVC) status

.PHONY: pipeline
pipeline: ## Reproduce the complete DVC pipeline
	$(DVC) repro

.PHONY: pipeline-force
pipeline-force: ## Force reproduction of the complete DVC pipeline
	$(DVC) repro --force

.PHONY: mlflow-local
mlflow-local: ## Start a local MLflow tracking server
	$(RUN) mlflow server \
		--backend-store-uri $(MLFLOW_BACKEND_STORE_URI) \
		--serve-artifacts \
		--artifacts-destination $(MLFLOW_ARTIFACT_ROOT) \
		--host $(MLFLOW_HOST) \
		--port $(MLFLOW_PORT) \
		--workers $(MLFLOW_WORKERS)

.PHONY: docker-build
docker-build: ## Build the runtime and pipeline Docker images
	docker build --target runtime -t retailrocket-recommender:runtime .
	docker build --target pipeline -t retailrocket-recommender:pipeline .

.PHONY: docker-runtime
docker-runtime: ## Run the runtime image
	docker run --rm retailrocket-recommender:runtime

.PHONY: docker-pipeline
docker-pipeline: ## Run the default command of the pipeline image
	docker run --rm retailrocket-recommender:pipeline

.PHONY: compose-config
compose-config: ## Validate the Docker Compose configuration
	$(COMPOSE) config --quiet

.PHONY: compose-build
compose-build: ## Build the Docker Compose services
	$(COMPOSE) build

.PHONY: compose-up
compose-up: ## Start the MLflow service
	$(COMPOSE) up -d mlflow

.PHONY: compose-status
compose-status: ## Display Docker Compose service status
	$(COMPOSE) ps

.PHONY: compose-logs
compose-logs: ## Follow the MLflow service logs
	$(COMPOSE) logs -f mlflow

.PHONY: compose-down
compose-down: ## Stop Docker Compose services and preserve volumes
	$(COMPOSE) down

.PHONY: compose-dag
compose-dag: ## Display the DVC DAG inside the trainer container
	$(COMPOSE) run --rm trainer poetry run dvc dag

.PHONY: compose-status-dvc
compose-status-dvc: ## Check DVC status inside the trainer container
	$(COMPOSE) run --rm trainer poetry run dvc status

.PHONY: promote-model compose-promote-model
compose-promote-model: ## Promote the selected model to production through Docker Compose
	$(COMPOSE) run --rm trainer poetry run python -m retail_recommender.pipelines.register_model --promote-to-production

.PHONY: compose-pipeline
compose-pipeline: ## Reproduce the DVC pipeline inside the trainer container
	$(COMPOSE) run --rm trainer poetry run dvc repro

.PHONY: compose-register-model
compose-register-model: ## Register the selected model through Docker Compose
	$(COMPOSE) run --rm trainer poetry run python \
		-m retail_recommender.pipelines.register_model

.PHONY: compose-test
compose-test: ## Run the complete test suite inside the trainer container
	$(COMPOSE) run --rm trainer poetry run pytest

.PHONY: compose-quality
compose-quality: ## Run lint, formatting and tests inside the trainer container
	$(COMPOSE) run --rm trainer poetry run ruff check .
	$(COMPOSE) run --rm trainer poetry run ruff format --check .
	$(COMPOSE) run --rm trainer poetry run pytest

.PHONY: validate-local
validate-local: check-dependencies quality dvc-status ## Run all local validations

.PHONY: validate-compose
validate-compose: compose-config compose-quality compose-status-dvc ## Run Compose validations

.PHONY: validate-all
validate-all: validate-local validate-compose ## Run all local and container validations
