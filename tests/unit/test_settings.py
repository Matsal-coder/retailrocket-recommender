"""Tests for application settings."""

from pathlib import Path

import pytest

from retail_recommender.config.settings import Settings


def test_settings_has_default_values(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Settings should provide safe default values."""
    monkeypatch.delenv("APP_ENV", raising=False)
    settings = Settings(_env_file=None)

    assert settings.app_name == "retailrocket-recommender"
    assert settings.app_env == "local"
    assert settings.mlflow_registered_model_name == "RetailRocketRecommender"
    assert settings.mlflow_staging_alias == "staging"
    assert settings.mlflow_production_alias == "production"


def test_settings_reads_environment_variables(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("APP_ENV", "dev")

    settings = Settings(_env_file=None)

    assert settings.app_env == "dev"


def test_settings_uses_path_objects() -> None:
    """Directory settings should be represented as Path objects."""
    settings = Settings()

    assert isinstance(settings.data_dir, Path)
    assert isinstance(settings.raw_data_dir, Path)
    assert isinstance(settings.processed_data_dir, Path)
