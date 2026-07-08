"""Tests for logging configuration."""

import logging

import pytest

from retail_recommender.config.logging import configure_logging, get_logger


def test_configure_logging_sets_expected_level() -> None:
    """Logging configuration should set the root logger level."""
    configure_logging("INFO")

    assert logging.getLogger().level == logging.INFO


def test_configure_logging_rejects_invalid_level() -> None:
    """Invalid log levels should raise ValueError."""
    with pytest.raises(ValueError, match="Invalid log level"):
        configure_logging("INVALID")


def test_get_logger_returns_logger_instance() -> None:
    """get_logger should return a logging.Logger instance."""
    logger = get_logger("test_logger")

    assert isinstance(logger, logging.Logger)
    assert logger.name == "test_logger"