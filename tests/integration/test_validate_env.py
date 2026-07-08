"""Integration tests for environment validation."""

from retail_recommender.config.settings import Settings
from scripts.validate_env import validate_directories


def test_validate_directories_with_project_defaults() -> None:
    """Default project directories should exist in the repository."""
    settings = Settings()

    validate_directories(settings)