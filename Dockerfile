# syntax=docker/dockerfile:1

ARG PYTHON_VERSION=3.12
ARG POETRY_VERSION=2.4.1


FROM python:${PYTHON_VERSION}-slim AS poetry-base

ARG POETRY_VERSION

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1 \
    POETRY_NO_INTERACTION=1 \
    POETRY_VIRTUALENVS_CREATE=false \
    POETRY_HOME=/opt/poetry \
    VIRTUAL_ENV=/opt/venv \
    PATH="/opt/venv/bin:/opt/poetry/bin:${PATH}"

WORKDIR /app

RUN apt-get update \
    && apt-get install --no-install-recommends --yes \
        build-essential \
        curl \
    && rm -rf /var/lib/apt/lists/*

RUN python -m venv "${POETRY_HOME}" \
    && "${POETRY_HOME}/bin/pip" install \
        --upgrade \
        pip \
        setuptools \
        wheel \
    && "${POETRY_HOME}/bin/pip" install \
        "poetry==${POETRY_VERSION}"

COPY pyproject.toml poetry.lock ./


FROM poetry-base AS dependency-builder

RUN python -m venv "${VIRTUAL_ENV}" \
    && poetry install \
        --only main \
        --no-root \
        --no-ansi


FROM poetry-base AS pipeline-builder

RUN python -m venv "${VIRTUAL_ENV}" \
    && poetry install \
        --with dev \
        --no-root \
        --no-ansi


FROM python:${PYTHON_VERSION}-slim AS runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    POETRY_NO_INTERACTION=1 \
    POETRY_VIRTUALENVS_CREATE=false \
    POETRY_HOME=/opt/poetry \
    VIRTUAL_ENV=/opt/venv \
    PATH="/opt/venv/bin:/opt/poetry/bin:${PATH}" \
    PYTHONPATH=/app/src

WORKDIR /app

RUN apt-get update \
    && apt-get install --no-install-recommends --yes \
        libgomp1 \
    && rm -rf /var/lib/apt/lists/* \
    && groupadd --system app \
    && useradd \
        --system \
        --gid app \
        --create-home \
        app

COPY --from=poetry-base /opt/poetry /opt/poetry
COPY --from=dependency-builder /opt/venv /opt/venv

COPY pyproject.toml poetry.lock ./
COPY src ./src
COPY configs ./configs
COPY scripts ./scripts
COPY params.yaml ./
COPY .env.example ./

RUN mkdir -p \
        data/raw \
        data/interim \
        data/processed \
        artifacts/models \
        artifacts/encoders \
        artifacts/reports/evaluation \
        artifacts/reports/registry \
    && chown -R app:app /app

USER app

CMD ["poetry", "run", "python", "scripts/validate_env.py"]


FROM python:${PYTHON_VERSION}-slim AS pipeline

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    POETRY_NO_INTERACTION=1 \
    POETRY_VIRTUALENVS_CREATE=false \
    POETRY_HOME=/opt/poetry \
    VIRTUAL_ENV=/opt/venv \
    PATH="/opt/venv/bin:/opt/poetry/bin:${PATH}" \
    PYTHONPATH=/app/src

WORKDIR /app

RUN apt-get update \
    && apt-get install --no-install-recommends --yes \
        git \
        libgomp1 \
    && rm -rf /var/lib/apt/lists/* \
    && groupadd --system app \
    && useradd \
        --system \
        --gid app \
        --create-home \
        app

COPY --from=poetry-base /opt/poetry /opt/poetry
COPY --from=pipeline-builder /opt/venv /opt/venv

COPY pyproject.toml poetry.lock ./
COPY src ./src
COPY configs ./configs
COPY scripts ./scripts
COPY tests ./tests
COPY params.yaml ./
COPY .env.example ./
COPY dvc.yaml dvc.lock ./
COPY .dvc ./.dvc
COPY data ./data

RUN mkdir -p \
        artifacts/models \
        artifacts/encoders \
        artifacts/reports/evaluation \
        artifacts/reports/registry \
    && chown -R app:app /app

USER app

CMD ["poetry", "run", "dvc", "--version"]
