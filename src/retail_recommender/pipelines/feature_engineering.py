"""Build processed recommendation datasets from clean events."""

from __future__ import annotations

import argparse
import json
import logging
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

from retail_recommender.features.id_encoder import (
    IdEncoder,
    fit_interaction_encoders,
    transform_interaction_ids,
)
from retail_recommender.features.interaction_builder import (
    InteractionFilterConfig,
    build_and_filter_interactions,
)
from retail_recommender.features.negative_sampling import (
    NegativeSamplingConfig,
    combine_positive_and_negative_interactions,
    generate_negative_samples,
)
from retail_recommender.features.temporal_split import (
    TemporalSplitConfig,
    temporal_split,
)

LOGGER = logging.getLogger(__name__)

DEFAULT_DATA_CONFIG_PATH = Path("configs/data.yaml")
DEFAULT_PARAMS_PATH = Path("params.yaml")

TRAINING_COLUMNS = [
    "user_idx",
    "item_idx",
    "target",
]

EVALUATION_COLUMNS = [
    "user_id",
    "item_id",
    "user_idx",
    "item_idx",
    "interaction_score",
    "interaction_count",
    "last_interaction_at",
    "target",
]


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
        msg = f"Expected a mapping at the root of: {path}"
        raise ValueError(msg)

    return content


def get_required_mapping(
    config: Mapping[str, Any],
    key: str,
) -> Mapping[str, Any]:
    """Return a required configuration mapping."""
    value = config.get(key)

    if not isinstance(value, Mapping):
        msg = f"Missing or invalid configuration section: {key}"
        raise ValueError(msg)

    return value


def get_required_path(
    config: Mapping[str, Any],
    key: str,
) -> Path:
    """Return a required path configuration."""
    value = config.get(key)

    if not isinstance(value, str) or not value.strip():
        msg = f"Missing or invalid path configuration: {key}"
        raise ValueError(msg)

    return Path(value)


def get_required_int(
    config: Mapping[str, Any],
    key: str,
) -> int:
    """Return a required integer configuration."""
    value = config.get(key)

    if not isinstance(value, int):
        msg = f"Missing or invalid integer parameter: {key}"
        raise ValueError(msg)

    return value


def get_required_float(
    config: Mapping[str, Any],
    key: str,
) -> float:
    """Return a required numeric configuration as float."""
    value = config.get(key)

    if not isinstance(value, int | float):
        msg = f"Missing or invalid numeric parameter: {key}"
        raise ValueError(msg)

    return float(value)


def load_clean_events(path: Path) -> pd.DataFrame:
    """Load the clean event-level Parquet dataset."""
    if not path.exists():
        msg = f"Clean events file not found: {path}"
        raise FileNotFoundError(msg)

    events = pd.read_parquet(path)

    if events.empty:
        msg = f"Clean events file is empty: {path}"
        raise ValueError(msg)

    return events


def save_parquet(data: pd.DataFrame, path: Path) -> None:
    """Persist a DataFrame as Parquet."""
    path.parent.mkdir(parents=True, exist_ok=True)
    data.to_parquet(path, index=False)


def save_report(report: Mapping[str, Any], path: Path) -> None:
    """Persist a feature-engineering report as JSON."""
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8") as file:
        json.dump(report, file, indent=2, ensure_ascii=False)


def build_feature_engineering_report(
    clean_events: pd.DataFrame,
    interactions: pd.DataFrame,
    train_positive: pd.DataFrame,
    train_final: pd.DataFrame,
    validation: pd.DataFrame,
    test: pd.DataFrame,
    validation_removed_unknowns: int,
    test_removed_unknowns: int,
    user_encoder: IdEncoder,
    item_encoder: IdEncoder,
) -> dict[str, Any]:
    """Build summary statistics for feature engineering outputs."""
    return {
        "clean_event_count": len(clean_events),
        "interaction_count": len(interactions),
        "train_positive_count": len(train_positive),
        "train_final_count": len(train_final),
        "train_negative_count": int(train_final["target"].eq(0).sum()),
        "validation_count": len(validation),
        "test_count": len(test),
        "validation_removed_unknowns": validation_removed_unknowns,
        "test_removed_unknowns": test_removed_unknowns,
        "user_encoder_size": user_encoder.size,
        "item_encoder_size": item_encoder.size,
        "train_period": {
            "start": _datetime_to_string(train_positive["last_interaction_at"].min()),
            "end": _datetime_to_string(train_positive["last_interaction_at"].max()),
        },
        "validation_period": {
            "start": _datetime_to_string(validation["last_interaction_at"].min()),
            "end": _datetime_to_string(validation["last_interaction_at"].max()),
        },
        "test_period": {
            "start": _datetime_to_string(test["last_interaction_at"].min()),
            "end": _datetime_to_string(test["last_interaction_at"].max()),
        },
    }


def _datetime_to_string(value: Any) -> str | None:
    """Convert a datetime-like value to an ISO string."""
    if pd.isna(value):
        return None

    return pd.Timestamp(value).isoformat()


def run_feature_engineering_pipeline(
    data_config_path: Path = DEFAULT_DATA_CONFIG_PATH,
    params_path: Path = DEFAULT_PARAMS_PATH,
) -> dict[str, Any]:
    """Run the complete feature-engineering pipeline."""
    data_config = load_yaml_file(data_config_path)
    params = load_yaml_file(params_path)

    filtering_config = get_required_mapping(
        params,
        "interaction_filtering",
    )
    split_config = get_required_mapping(params, "split")
    training_config = get_required_mapping(params, "training")

    clean_events_path = get_required_path(
        data_config,
        "interim_events_path",
    )
    train_path = get_required_path(
        data_config,
        "train_data_path",
    )
    validation_path = get_required_path(
        data_config,
        "validation_data_path",
    )
    test_path = get_required_path(
        data_config,
        "test_data_path",
    )
    user_encoder_path = get_required_path(
        data_config,
        "user_encoder_path",
    )
    item_encoder_path = get_required_path(
        data_config,
        "item_encoder_path",
    )
    report_path = get_required_path(
        data_config,
        "feature_engineering_report_path",
    )

    LOGGER.info("Loading clean events from %s", clean_events_path)
    clean_events = load_clean_events(clean_events_path)

    LOGGER.info("Building and filtering user-item interactions")
    interactions = build_and_filter_interactions(
        events=clean_events,
        config=InteractionFilterConfig(
            minimum_user_interactions=get_required_int(
                filtering_config,
                "minimum_user_interactions",
            ),
            minimum_item_interactions=get_required_int(
                filtering_config,
                "minimum_item_interactions",
            ),
        ),
    )

    LOGGER.info("Splitting interactions chronologically")
    split_result = temporal_split(
        interactions=interactions,
        config=TemporalSplitConfig(
            train_size=get_required_float(
                split_config,
                "train_size",
            ),
            validation_size=get_required_float(
                split_config,
                "validation_size",
            ),
            test_size=get_required_float(
                split_config,
                "test_size",
            ),
            filter_unknown_entities=bool(
                split_config.get("filter_unknown_entities", True)
            ),
        ),
    )

    if split_result.validation.empty:
        msg = "Validation set is empty after temporal filtering."
        raise ValueError(msg)

    if split_result.test.empty:
        msg = "Test set is empty after temporal filtering."
        raise ValueError(msg)

    LOGGER.info("Fitting user and item encoders on training only")
    user_encoder, item_encoder = fit_interaction_encoders(split_result.train)

    train_positive = transform_interaction_ids(
        interactions=split_result.train,
        user_encoder=user_encoder,
        item_encoder=item_encoder,
        drop_unknowns=True,
    )
    validation = transform_interaction_ids(
        interactions=split_result.validation,
        user_encoder=user_encoder,
        item_encoder=item_encoder,
        drop_unknowns=True,
    )
    test = transform_interaction_ids(
        interactions=split_result.test,
        user_encoder=user_encoder,
        item_encoder=item_encoder,
        drop_unknowns=True,
    )

    random_seed = get_required_int(training_config, "random_seed")
    negative_samples_per_positive = get_required_int(
        training_config,
        "negative_samples_per_positive",
    )

    LOGGER.info("Generating negative training samples")
    negatives = generate_negative_samples(
        positive_interactions=train_positive,
        item_count=item_encoder.size,
        config=NegativeSamplingConfig(
            negative_samples_per_positive=(negative_samples_per_positive),
            random_seed=random_seed,
        ),
    )

    train_final = combine_positive_and_negative_interactions(
        positive_interactions=train_positive,
        negative_interactions=negatives,
        random_seed=random_seed,
    )

    LOGGER.info("Saving processed datasets")
    save_parquet(
        train_final.loc[:, TRAINING_COLUMNS],
        train_path,
    )
    save_parquet(
        validation.loc[:, EVALUATION_COLUMNS],
        validation_path,
    )
    save_parquet(
        test.loc[:, EVALUATION_COLUMNS],
        test_path,
    )

    LOGGER.info("Saving fitted encoders")
    user_encoder.save(user_encoder_path)
    item_encoder.save(item_encoder_path)

    report = build_feature_engineering_report(
        clean_events=clean_events,
        interactions=interactions,
        train_positive=train_positive,
        train_final=train_final,
        validation=validation,
        test=test,
        validation_removed_unknowns=(split_result.validation_removed_unknowns),
        test_removed_unknowns=split_result.test_removed_unknowns,
        user_encoder=user_encoder,
        item_encoder=item_encoder,
    )

    save_report(report, report_path)

    LOGGER.info("Feature-engineering pipeline completed")

    return report


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Build processed recommendation datasets.",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_DATA_CONFIG_PATH,
    )
    parser.add_argument(
        "--params",
        type=Path,
        default=DEFAULT_PARAMS_PATH,
    )

    return parser.parse_args()


def main() -> None:
    """Run feature engineering from the command line."""
    args = parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )

    run_feature_engineering_pipeline(
        data_config_path=args.config,
        params_path=args.params,
    )


if __name__ == "__main__":
    main()
