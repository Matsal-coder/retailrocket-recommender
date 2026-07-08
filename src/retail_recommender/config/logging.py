"""Logging configuration for the application."""

import logging
import sys

DEFAULT_LOG_FORMAT = "%(asctime)s | %(levelname)s | %(name)s | %(message)s"


def configure_logging(log_level: str = "INFO") -> None:
    """Configure application logging.

    Args:
        log_level: Minimum logging level, such as "DEBUG", "INFO" or "WARNING".
    """
    normalized_level = log_level.upper()
    numeric_level = logging.getLevelName(normalized_level)

    if not isinstance(numeric_level, int):
        msg = f"Invalid log level: {log_level}"
        raise ValueError(msg)

    logging.basicConfig(
        level=numeric_level,
        format=DEFAULT_LOG_FORMAT,
        handlers=[logging.StreamHandler(sys.stdout)],
        force=True,
    )


def get_logger(name: str) -> logging.Logger:
    """Return a logger using the project's logging convention.

    Args:
        name: Logger name, usually `__name__`.

    Returns:
        Configured logger instance.
    """
    return logging.getLogger(name)
