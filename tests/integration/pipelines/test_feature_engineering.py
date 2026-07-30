"""Integration tests for the feature-engineering pipeline."""

from pathlib import Path

import pandas as pd
import yaml

from retail_recommender.features.id_encoder import IdEncoder
from retail_recommender.pipelines.feature_engineering import (
    run_feature_engineering_pipeline,
)

RANDOM_SEED = 1729
NEGATIVES_PER_POSITIVE = 2

EXPECTED_TRAIN_COLUMNS = [
    "user_idx",
    "item_idx",
    "target",
]

EXPECTED_EVALUATION_COLUMNS = [
    "user_id",
    "item_id",
    "user_idx",
    "item_idx",
    "interaction_score",
    "interaction_count",
    "last_interaction_at",
    "target",
]


def make_clean_events() -> pd.DataFrame:
    """Create clean events with known future users and items."""
    interaction_pairs = [
        (10, 100),
        (10, 200),
        (10, 300),
        (10, 400),
        (10, 500),
        (20, 100),
        (20, 200),
        (20, 300),
        (20, 600),
        (20, 700),
        (30, 100),
        (30, 300),
        (30, 400),
        (30, 700),
        (30, 800),
        (40, 200),
        (40, 400),
        (40, 500),
        (40, 600),
        (40, 800),
        (10, 600),
        (20, 800),
        (30, 500),
        (40, 700),
        (10, 700),
        (20, 400),
        (30, 600),
        (40, 100),
        (10, 800),
        (20, 500),
    ]

    row_count = len(interaction_pairs)

    return pd.DataFrame(
        {
            "user_id": [user_id for user_id, _ in interaction_pairs],
            "item_id": [item_id for _, item_id in interaction_pairs],
            "event_type": ["view"] * row_count,
            "event_weight": [1.0] * row_count,
            "timestamp": [
                1_600_000_000_000 + index * 86_400_000 for index in range(row_count)
            ],
            "datetime": pd.date_range(
                start="2026-01-01",
                periods=row_count,
                freq="D",
                tz="UTC",
            ),
        }
    )


def make_params() -> dict[str, object]:
    """Create feature-engineering test parameters."""
    return {
        "interaction_filtering": {
            "minimum_user_interactions": 1,
            "minimum_item_interactions": 1,
        },
        "split": {
            "strategy": "temporal",
            "train_size": 0.70,
            "validation_size": 0.15,
            "test_size": 0.15,
            "filter_unknown_entities": True,
        },
        "training": {
            "negative_samples_per_positive": NEGATIVES_PER_POSITIVE,
            "random_seed": RANDOM_SEED,
        },
    }


def make_data_config(
    interim_path: Path,
    train_path: Path,
    positive_train_path: Path,
    validation_path: Path,
    test_path: Path,
    user_encoder_path: Path,
    item_encoder_path: Path,
    report_path: Path,
) -> dict[str, str]:
    """Create temporary pipeline path configuration."""
    return {
        "interim_events_path": str(interim_path),
        "train_data_path": str(train_path),
        "train_positive_path": str(positive_train_path),
        "validation_data_path": str(validation_path),
        "test_data_path": str(test_path),
        "user_encoder_path": str(user_encoder_path),
        "item_encoder_path": str(item_encoder_path),
        "feature_engineering_report_path": str(report_path),
    }


def assert_output_files_exist(paths: list[Path]) -> None:
    """Assert that all expected pipeline outputs exist."""
    assert all(path.exists() for path in paths)


def assert_non_negative_indices(data: pd.DataFrame) -> None:
    """Assert that encoded user and item indices are valid."""
    assert data["user_idx"].ge(0).all()
    assert data["item_idx"].ge(0).all()


def test_feature_engineering_pipeline_generates_outputs(
    tmp_path: Path,
) -> None:
    """It should generate processed splits, encoders and report."""
    interim_path = tmp_path / "data" / "interim" / "events_clean.parquet"
    train_path = tmp_path / "data" / "processed" / "train.parquet"
    positive_train_path = tmp_path / "data" / "processed" / "train_positive.parquet"
    validation_path = tmp_path / "data" / "processed" / "validation.parquet"
    test_path = tmp_path / "data" / "processed" / "test.parquet"

    user_encoder_path = tmp_path / "artifacts" / "encoders" / "user_encoder.pkl"
    item_encoder_path = tmp_path / "artifacts" / "encoders" / "item_encoder.pkl"
    report_path = tmp_path / "artifacts" / "reports" / "feature_engineering_report.json"

    config_path = tmp_path / "configs" / "data.yaml"
    params_path = tmp_path / "params.yaml"

    interim_path.parent.mkdir(parents=True)
    config_path.parent.mkdir(parents=True)

    clean_events = make_clean_events()
    clean_events.to_parquet(interim_path, index=False)

    data_config = make_data_config(
        interim_path=interim_path,
        train_path=train_path,
        positive_train_path=positive_train_path,
        validation_path=validation_path,
        test_path=test_path,
        user_encoder_path=user_encoder_path,
        item_encoder_path=item_encoder_path,
        report_path=report_path,
    )

    params = make_params()

    config_path.write_text(
        yaml.safe_dump(data_config),
        encoding="utf-8",
    )
    params_path.write_text(
        yaml.safe_dump(params),
        encoding="utf-8",
    )

    report = run_feature_engineering_pipeline(
        data_config_path=config_path,
        params_path=params_path,
    )

    assert_output_files_exist(
        [
            train_path,
            positive_train_path,
            validation_path,
            test_path,
            user_encoder_path,
            item_encoder_path,
            report_path,
        ]
    )

    train = pd.read_parquet(train_path)
    validation = pd.read_parquet(validation_path)
    test = pd.read_parquet(test_path)
    positive_train = pd.read_parquet(positive_train_path)

    assert train.columns.tolist() == EXPECTED_TRAIN_COLUMNS
    assert validation.columns.tolist() == EXPECTED_EVALUATION_COLUMNS
    assert test.columns.tolist() == EXPECTED_EVALUATION_COLUMNS
    assert not train.empty
    assert not validation.empty
    assert not test.empty
    assert not positive_train.empty

    assert set(train["target"].unique()) == {0, 1}
    assert set(positive_train["target"]) == {1}
    assert {
        "interaction_score",
        "interaction_count",
    }.issubset(positive_train.columns)
    assert validation["target"].eq(1).all()
    assert test["target"].eq(1).all()

    assert_non_negative_indices(train)
    assert_non_negative_indices(validation)
    assert_non_negative_indices(test)

    user_encoder = IdEncoder.load(user_encoder_path)
    item_encoder = IdEncoder.load(item_encoder_path)

    assert user_encoder.size > 0
    assert item_encoder.size > 0
    assert report["train_negative_count"] > 0

    train_positive_count = train["target"].eq(1).sum()
    train_negative_count = train["target"].eq(0).sum()

    assert train_negative_count == train_positive_count * NEGATIVES_PER_POSITIVE
