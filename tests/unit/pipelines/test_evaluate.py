"""Unit tests for evaluation pipeline helpers."""

from pathlib import Path

import pandas as pd
import pytest
import yaml

from retail_recommender.pipelines import evaluate as evaluate_pipeline
from retail_recommender.pipelines.evaluate import (
    BASELINE_REQUIRED_COLUMNS,
    TEST_REQUIRED_COLUMNS,
    _build_user_item_mapping,
    _positive_interactions,
    _resolve_entity_counts,
    _validate_interactions,
)

EXPECTED_USER_COUNT = 4
EXPECTED_ITEM_COUNT = 6


def test_load_evaluation_paths_uses_centralized_configs(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    train_path = tmp_path / "train_positive.parquet"
    test_path = tmp_path / "test.parquet"
    checkpoint_path = tmp_path / "best_model.pt"
    report_path = tmp_path / "train_metrics.json"
    output_directory = tmp_path / "evaluation"

    data_config_path = tmp_path / "data.yaml"
    model_config_path = tmp_path / "model.yaml"
    training_config_path = tmp_path / "training.yaml"
    evaluation_config_path = tmp_path / "evaluation.yaml"

    data_config_path.write_text(
        yaml.safe_dump(
            {
                "train_positive_path": str(train_path),
                "test_data_path": str(test_path),
            }
        ),
        encoding="utf-8",
    )
    model_config_path.write_text(
        yaml.safe_dump(
            {
                "model": {
                    "checkpoint_path": str(checkpoint_path),
                }
            }
        ),
        encoding="utf-8",
    )
    training_config_path.write_text(
        yaml.safe_dump(
            {
                "training": {
                    "metrics_report_path": str(report_path),
                }
            }
        ),
        encoding="utf-8",
    )
    evaluation_config_path.write_text(
        yaml.safe_dump(
            {
                "evaluation": {
                    "output_directory": str(output_directory),
                }
            }
        ),
        encoding="utf-8",
    )

    monkeypatch.setattr(
        evaluate_pipeline,
        "DATA_CONFIG_PATH",
        data_config_path,
    )
    monkeypatch.setattr(
        evaluate_pipeline,
        "MODEL_CONFIG_PATH",
        model_config_path,
    )
    monkeypatch.setattr(
        evaluate_pipeline,
        "TRAINING_CONFIG_PATH",
        training_config_path,
    )
    monkeypatch.setattr(
        evaluate_pipeline,
        "EVALUATION_CONFIG_PATH",
        evaluation_config_path,
    )

    paths = evaluate_pipeline._load_evaluation_paths()

    assert paths.train_interactions == train_path
    assert paths.test_data == test_path
    assert paths.checkpoint == checkpoint_path
    assert paths.training_report == report_path
    assert paths.output_directory == output_directory


def test_positive_interactions_filters_negative_samples() -> None:
    interactions = pd.DataFrame(
        {
            "user_idx": [0, 0, 1],
            "item_idx": [1, 2, 3],
            "target": [1, 0, 1],
        }
    )

    positive = _positive_interactions(interactions)

    assert positive["item_idx"].tolist() == [1, 3]


def test_build_user_item_mapping_groups_items() -> None:
    interactions = pd.DataFrame(
        {
            "user_idx": [0, 0, 1],
            "item_idx": [1, 2, 3],
            "target": [1, 1, 1],
        }
    )

    mapping = _build_user_item_mapping(interactions)

    assert mapping == {
        0: {1, 2},
        1: {3},
    }


def test_resolve_entity_counts_uses_maximum_ids() -> None:
    train_interactions = pd.DataFrame(
        {
            "user_idx": [0, 2],
            "item_idx": [0, 4],
            "target": [1, 1],
        }
    )
    test_interactions = pd.DataFrame(
        {
            "user_idx": [3],
            "item_idx": [5],
            "target": [1],
        }
    )

    num_users, num_items = _resolve_entity_counts(
        train_interactions,
        test_interactions,
    )

    assert num_users == EXPECTED_USER_COUNT
    assert num_items == EXPECTED_ITEM_COUNT


def test_validate_interactions_rejects_missing_target() -> None:
    interactions = pd.DataFrame(
        {
            "user_idx": [0],
            "item_idx": [1],
        }
    )

    with pytest.raises(
        ValueError,
        match="test data is missing columns: target",
    ):
        _validate_interactions(
            interactions,
            split_name="test",
            required_columns=TEST_REQUIRED_COLUMNS,
        )


def test_validate_interactions_rejects_missing_baseline_columns() -> None:
    interactions = pd.DataFrame(
        {
            "user_idx": [0],
            "item_idx": [1],
            "target": [1],
        }
    )

    with pytest.raises(
        ValueError,
        match=(
            "training data is missing columns: " "interaction_count, interaction_score"
        ),
    ):
        _validate_interactions(
            interactions,
            split_name="training",
            required_columns=BASELINE_REQUIRED_COLUMNS,
        )
