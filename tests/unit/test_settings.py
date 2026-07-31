"""Tests for application settings."""

from pathlib import Path

from retail_recommender.config.settings import Settings


def test_settings_has_default_values() -> None:
    """Settings should provide safe default values."""
    settings = Settings()

    assert settings.app_name == "retailrocket-recommender"
    assert settings.app_env == "local"


def test_settings_uses_path_objects() -> None:
    """Directory settings should be represented as Path objects."""
    settings = Settings()

    assert isinstance(settings.data_dir, Path)
    assert isinstance(settings.raw_data_dir, Path)
    assert isinstance(settings.processed_data_dir, Path)
