"""Unit tests for the model registration pipeline."""

import json
from pathlib import Path

import pandas as pd
import pytest
import yaml

from retail_recommender.pipelines import (
    register_model as register_model_pipeline,
)
from retail_recommender.pipelines.register_model import (
    RegistrationResult,
    _build_version_tags,
    _load_json,
    _validate_train_interactions,
    _write_registration_report,
)

EXPECTED_K = 10
EXPECTED_USERS = 50


def build_selection() -> dict[str, object]:
    """Create valid selected-model metadata."""

    return {
        "model_name": "item_knn",
        "primary_metric": "ndcg_at_k",
        "primary_metric_value": 0.04,
        "k": EXPECTED_K,
        "evaluated_users": EXPECTED_USERS,
    }


def test_build_version_tags_uses_selection_metadata() -> None:
    tags = _build_version_tags(build_selection())

    assert tags == {
        "model_type": "item_knn",
        "primary_metric": "ndcg_at_k",
        "primary_metric_value": "0.04",
        "k": "10",
        "evaluated_users": "50",
        "validation_status": "passed",
    }


def test_validate_train_interactions_accepts_valid_data() -> None:
    interactions = pd.DataFrame(
        {
            "user_idx": [0],
            "item_idx": [1],
            "interaction_score": [1.0],
            "interaction_count": [1],
            "target": [1],
        }
    )

    _validate_train_interactions(interactions)


def test_validate_train_interactions_rejects_missing_columns() -> None:
    interactions = pd.DataFrame(
        {
            "user_idx": [0],
            "item_idx": [1],
        }
    )

    with pytest.raises(ValueError, match="missing columns"):
        _validate_train_interactions(interactions)


def test_validate_train_interactions_rejects_empty_data() -> None:
    interactions = pd.DataFrame(
        columns=[
            "user_idx",
            "item_idx",
            "interaction_score",
            "interaction_count",
            "target",
        ]
    )

    with pytest.raises(ValueError, match="cannot be empty"):
        _validate_train_interactions(interactions)


def test_load_json_rejects_missing_file(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        _load_json(tmp_path / "missing.json")


def test_write_registration_report_writes_json(
    tmp_path: Path,
) -> None:
    path = tmp_path / "registry" / "registration.json"
    result = RegistrationResult(
        registered_model_name="RetailRocketRecommender",
        version="1",
        alias="staging",
        source_model="item_knn",
        model_uri="models:/m-test",
        run_id="run-test",
        primary_metric="ndcg_at_k",
        primary_metric_value=0.04,
        k=EXPECTED_K,
        evaluated_users=EXPECTED_USERS,
    )

    _write_registration_report(
        result=result,
        path=path,
    )

    saved = json.loads(path.read_text(encoding="utf-8"))

    assert saved["registered_model_name"] == ("RetailRocketRecommender")
    assert saved["version"] == "1"
    assert saved["alias"] == "staging"


def test_load_registration_paths_uses_centralized_configs(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    train_path = tmp_path / "train_positive.parquet"
    selected_model_path = tmp_path / "selected_model.json"
    registry_report_path = tmp_path / "registry" / "registration.json"

    data_config_path = tmp_path / "data.yaml"
    evaluation_config_path = tmp_path / "evaluation.yaml"
    registry_config_path = tmp_path / "registry.yaml"

    data_config_path.write_text(
        yaml.safe_dump(
            {
                "train_positive_path": str(train_path),
            }
        ),
        encoding="utf-8",
    )
    evaluation_config_path.write_text(
        yaml.safe_dump(
            {
                "evaluation": {
                    "selected_model_path": str(selected_model_path),
                }
            }
        ),
        encoding="utf-8",
    )
    registry_config_path.write_text(
        yaml.safe_dump(
            {
                "registry": {
                    "report_path": str(registry_report_path),
                }
            }
        ),
        encoding="utf-8",
    )

    monkeypatch.setattr(
        register_model_pipeline,
        "DATA_CONFIG_PATH",
        data_config_path,
    )
    monkeypatch.setattr(
        register_model_pipeline,
        "EVALUATION_CONFIG_PATH",
        evaluation_config_path,
    )
    monkeypatch.setattr(
        register_model_pipeline,
        "REGISTRY_CONFIG_PATH",
        registry_config_path,
    )

    paths = register_model_pipeline._load_registration_paths()

    assert paths.train_interactions == train_path
    assert paths.selected_model == selected_model_path
    assert paths.registry_report == registry_report_path
