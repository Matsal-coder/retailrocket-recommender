"""Preprocess raw RetailRocket events into weighted implicit feedback."""

from __future__ import annotations

import argparse
import logging
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

from retail_recommender.data.loaders.retailrocket_loader import load_events
from retail_recommender.data.preprocessors.factory import (
    EventPreprocessorFactory,
)

LOGGER = logging.getLogger(__name__)

DEFAULT_DATA_CONFIG_PATH = Path("configs/data.yaml")
DEFAULT_PARAMS_PATH = Path("params.yaml")


def load_yaml_file(path: Path) -> dict[str, Any]:
    """Load a YAML file as a dictionary.

    Args:
        path: YAML file path.

    Returns:
        Parsed YAML content.

    Raises:
        FileNotFoundError: If the YAML file does not exist.
        ValueError: If the YAML root is not a mapping.
    """
    if not path.exists():
        msg = f"Configuration file not found: {path}"
        raise FileNotFoundError(msg)

    with path.open(encoding="utf-8") as file:
        content = yaml.safe_load(file) or {}

    if not isinstance(content, dict):
        msg = f"Expected a mapping at the root of configuration file: {path}"
        raise ValueError(msg)

    return content


def get_required_mapping(
    config: Mapping[str, Any],
    key: str,
) -> Mapping[str, Any]:
    """Return a required configuration mapping.

    Args:
        config: Parent configuration mapping.
        key: Required configuration key.

    Returns:
        Mapping stored under the requested key.

    Raises:
        ValueError: If the key is missing or does not contain a mapping.
    """
    value = config.get(key)

    if not isinstance(value, Mapping):
        msg = f"Missing or invalid configuration section: {key}"
        raise ValueError(msg)

    return value


def load_raw_events(input_path: Path) -> pd.DataFrame:
    """Load raw RetailRocket events using the project loader.

    Args:
        input_path: Path to the raw RetailRocket events CSV.

    Returns:
        Raw RetailRocket events.
    """
    return load_events(input_path)


def preprocess_events(
    events: pd.DataFrame,
    preprocessing_config: Mapping[str, Any],
) -> pd.DataFrame:
    """Apply the configured event preprocessing strategy.

    Args:
        events: Raw RetailRocket events.
        preprocessing_config: Preprocessing parameters.

    Returns:
        Clean and standardized event data.
    """
    strategy = str(preprocessing_config.get("strategy", "")).strip()
    event_weights = get_required_mapping(preprocessing_config, "event_weights")

    allowed_event_types_value = preprocessing_config.get("allowed_event_types")
    if not isinstance(allowed_event_types_value, list):
        msg = "preprocessing.allowed_event_types must be a list."
        raise ValueError(msg)

    preprocessor = EventPreprocessorFactory.create(
        strategy=strategy,
        event_weights={
            str(event_type): float(weight)
            for event_type, weight in event_weights.items()
        },
        allowed_event_types=[
            str(event_type) for event_type in allowed_event_types_value
        ],
    )

    return preprocessor.transform(events)


def save_preprocessed_events(
    events: pd.DataFrame,
    output_path: Path,
) -> None:
    """Save preprocessed events as a Parquet file.

    Args:
        events: Preprocessed event data.
        output_path: Destination Parquet path.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    events.to_parquet(output_path, index=False)


def run_preprocessing_pipeline(
    data_config_path: Path = DEFAULT_DATA_CONFIG_PATH,
    params_path: Path = DEFAULT_PARAMS_PATH,
) -> pd.DataFrame:
    """Execute the complete event preprocessing pipeline.

    Args:
        data_config_path: Dataset configuration path.
        params_path: Pipeline parameters path.

    Returns:
        Preprocessed events DataFrame.
    """
    data_config = load_yaml_file(data_config_path)
    params = load_yaml_file(params_path)

    preprocessing_config = get_required_mapping(
        params,
        "preprocessing",
    )

    input_path = Path(str(data_config["raw_events_path"]))
    output_path = Path(str(data_config["interim_events_path"]))

    LOGGER.info("Loading raw events from %s", input_path)
    raw_events = load_raw_events(input_path)

    LOGGER.info("Preprocessing %s raw events", len(raw_events))
    clean_events = preprocess_events(raw_events, preprocessing_config)

    LOGGER.info(
        "Saving %s clean events to %s",
        len(clean_events),
        output_path,
    )
    save_preprocessed_events(clean_events, output_path)

    LOGGER.info("Preprocessing pipeline completed successfully")

    return clean_events


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Preprocess RetailRocket event data.",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_DATA_CONFIG_PATH,
        help="Path to the dataset YAML configuration.",
    )
    parser.add_argument(
        "--params",
        type=Path,
        default=DEFAULT_PARAMS_PATH,
        help="Path to the pipeline parameters YAML file.",
    )

    return parser.parse_args()


def main() -> None:
    """Run the preprocessing pipeline from the command line."""
    args = parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )

    run_preprocessing_pipeline(
        data_config_path=args.config,
        params_path=args.params,
    )


if __name__ == "__main__":
    main()
