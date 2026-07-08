"""Validate the local development environment."""

from pathlib import Path

from retail_recommender.config.logging import configure_logging, get_logger
from retail_recommender.config.settings import Settings, get_settings


def ensure_directory_exists(path: Path) -> None:
    """Raise an error if a required directory does not exist.

    Args:
        path: Directory path to validate.

    Raises:
        FileNotFoundError: If the path does not exist or is not a directory.
    """
    if not path.is_dir():
        msg = f"Required directory does not exist: {path}"
        raise FileNotFoundError(msg)


def validate_directories(settings: Settings) -> None:
    """Validate required project directories.

    Args:
        settings: Application settings.
    """
    required_directories = [
        settings.data_dir,
        settings.raw_data_dir,
        settings.interim_data_dir,
        settings.processed_data_dir,
        settings.artifacts_dir,
    ]

    for directory in required_directories:
        ensure_directory_exists(directory)


def main() -> None:
    """Run environment validation."""
    settings = get_settings()
    configure_logging(settings.log_level)

    logger = get_logger(__name__)
    logger.info("Validating environment for %s", settings.app_name)

    validate_directories(settings)

    logger.info("Environment validation finished successfully")


if __name__ == "__main__":
    main()
