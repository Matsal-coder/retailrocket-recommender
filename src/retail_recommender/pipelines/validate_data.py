"""Pipeline for validating RetailRocket raw events data."""

import argparse
import json
import logging
from pathlib import Path
from typing import Any

import yaml

from retail_recommender.config.logging import configure_logging
from retail_recommender.data.loaders.retailrocket_loader import load_events
from retail_recommender.data.validators.events_validator import (
    EventsValidationConfig,
    EventsValidationResult,
    validate_events,
)

DEFAULT_CONFIG_PATH = Path("configs/data.yaml")


def run_validate_data_pipeline(
    config_path: str | Path = DEFAULT_CONFIG_PATH,
) -> EventsValidationResult:
    """Run the raw events validation pipeline.

    Parameters
    ----------
    config_path:
        Path to the dataset validation configuration file.

    Returns
    -------
    EventsValidationResult
        Structured validation result.

    Raises
    ------
    ValueError
        If the dataset validation fails.
    """
    configure_logging()
    logger = logging.getLogger(__name__)

    config = _load_data_config(config_path)

    raw_events_path = Path(config["raw_events_path"])
    validation_report_path = Path(config["validation_report_path"])

    logger.info("Loading raw events from %s", raw_events_path)
    events = load_events(raw_events_path)

    logger.info("Validating raw events dataset")
    validation_result = validate_events(
        events,
        config=_build_validation_config(config),
    )

    logger.info("Saving validation report to %s", validation_report_path)
    _save_validation_report(validation_result, validation_report_path)

    if not validation_result.is_valid:
        errors = "; ".join(validation_result.errors)
        raise ValueError(f"Data validation failed: {errors}")

    logger.info("Data validation finished successfully")
    return validation_result


def _load_data_config(config_path: str | Path) -> dict[str, Any]:
    path = Path(config_path)

    if not path.exists():
        raise FileNotFoundError(f"Data config file not found: {path}")

    with path.open(encoding="utf-8") as config_file:
        config = yaml.safe_load(config_file)

    if not isinstance(config, dict):
        raise ValueError(f"Invalid data config format: {path}")

    return config


def _build_validation_config(config: dict[str, Any]) -> EventsValidationConfig:
    return EventsValidationConfig(
        required_columns=tuple(config["required_columns"]),
        allowed_events=frozenset(config["allowed_events"]),
        minimum_interactions=int(config["minimum_interactions"]),
        minimum_users=int(config["minimum_users"]),
        minimum_items=int(config["minimum_items"]),
    )


def _save_validation_report(
    validation_result: EventsValidationResult,
    output_path: str | Path,
) -> None:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8") as report_file:
        json.dump(
            validation_result.to_dict(), report_file, indent=2, ensure_ascii=False
        )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate RetailRocket raw events dataset.",
    )
    parser.add_argument(
        "--config",
        default=str(DEFAULT_CONFIG_PATH),
        help="Path to the data validation config YAML file.",
    )
    return parser.parse_args()


def main() -> None:
    """Run the validate data pipeline from the command line."""
    args = _parse_args()
    run_validate_data_pipeline(config_path=args.config)


if __name__ == "__main__":
    main()
