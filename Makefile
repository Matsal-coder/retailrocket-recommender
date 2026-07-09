.PHONY: install validate lint format test test-cov pre-commit clean

install:
	poetry install

validate:
	poetry run python scripts/validate_env.py

lint:
	poetry run ruff check .

format:
	poetry run ruff format .
	poetry run ruff check . --fix

test:
	poetry run pytest

test-cov:
	poetry run pytest --cov=retail_recommender --cov-report=term-missing

pre-commit:
	poetry run pre-commit run --all-files

clean:
	python -c "import shutil; from pathlib import Path; [shutil.rmtree(path, ignore_errors=True) for path in [Path('.pytest_cache'), Path('.ruff_cache'), Path('htmlcov')]]"
