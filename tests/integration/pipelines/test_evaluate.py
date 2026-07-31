"""Integration test for the evaluation pipeline."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

from retail_recommender.pipelines import evaluate as evaluate_pipeline

TEST_SEED = 1729
DEFAULT_K = 2
EXPECTED_MODEL_COUNT = 3
EXPECTED_EVALUATED_USERS = 2
NUM_USERS = 2
NUM_ITEMS = 4


class FakeRecommender:
    """Return deterministic recommendations."""

    def recommend(
        self,
        user_id: int,
        k: int,
        seen_items=None,
    ) -> list[int]:
        del user_id
        excluded_items = set(seen_items or ())
        candidates = [
            item_id for item_id in [3, 2, 1, 0] if item_id not in excluded_items
        ]
        return candidates[:k]


def _build_fake_models(**_: Any) -> dict[str, FakeRecommender]:
    """Return fake recommenders for pipeline integration testing."""
    return {
        "popularity": FakeRecommender(),
        "item_knn": FakeRecommender(),
        "neural_cf": FakeRecommender(),
    }


def test_evaluation_pipeline_generates_model_reports(
    tmp_path: Path,
    monkeypatch,
) -> None:
    train_interactions_path = tmp_path / "train_positive.parquet"
    test_path = tmp_path / "test.parquet"
    checkpoint_path = tmp_path / "best_model.pt"
    training_report_path = tmp_path / "train_metrics.json"
    output_directory = tmp_path / "evaluation"
    model_comparison_path = output_directory / "model_comparison.csv"
    selected_model_path = output_directory / "selected_model.json"

    params_path = tmp_path / "params.yaml"
    data_config_path = tmp_path / "data.yaml"
    model_config_path = tmp_path / "model.yaml"
    training_config_path = tmp_path / "training.yaml"
    evaluation_config_path = tmp_path / "evaluation.yaml"
    mlflow_database = tmp_path / "mlflow.db"

    train_interactions = pd.DataFrame(
        {
            "user_id": [10, 10, 20, 20],
            "item_id": [100, 200, 200, 300],
            "user_idx": [0, 0, 1, 1],
            "item_idx": [0, 1, 1, 2],
            "interaction_score": [1.0, 2.0, 1.0, 3.0],
            "interaction_count": [1, 2, 1, 2],
            "last_interaction_at": pd.to_datetime(
                [
                    "2025-01-01",
                    "2025-01-02",
                    "2025-01-03",
                    "2025-01-04",
                ],
                utc=True,
            ),
            "target": [1, 1, 1, 1],
        }
    )
    test_interactions = pd.DataFrame(
        {
            "user_id": [10, 20],
            "item_id": [300, 400],
            "user_idx": [0, 1],
            "item_idx": [2, 3],
            "interaction_score": [1.0, 1.0],
            "interaction_count": [1, 1],
            "last_interaction_at": pd.to_datetime(
                [
                    "2025-02-01",
                    "2025-02-02",
                ],
                utc=True,
            ),
            "target": [1, 1],
        }
    )

    train_interactions.to_parquet(
        train_interactions_path,
        index=False,
    )
    test_interactions.to_parquet(
        test_path,
        index=False,
    )
    checkpoint_path.write_bytes(b"test-checkpoint")
    training_report_path.write_text(
        json.dumps(
            {
                "model": {
                    "num_users": NUM_USERS,
                    "num_items": NUM_ITEMS,
                }
            }
        ),
        encoding="utf-8",
    )

    params_path.write_text(
        yaml.safe_dump(
            {
                "training": {
                    "random_seed": TEST_SEED,
                },
                "model": {
                    "name": "neural_cf",
                    "embedding_dim": 4,
                    "hidden_layers": [8, 4],
                    "dropout": 0.0,
                },
                "evaluation": {
                    "k": DEFAULT_K,
                    "candidate_batch_size": 4,
                    "exclude_seen_items": True,
                    "maximum_users": (EXPECTED_EVALUATED_USERS),
                },
                "item_knn": {
                    "n_neighbors": 2,
                    "minimum_similarity": 0.0,
                },
                "model_selection": {
                    "primary_metric": "ndcg_at_k",
                    "tie_breakers": [
                        "recall_at_k",
                        "map_at_k",
                        "coverage_at_k",
                    ],
                },
            }
        ),
        encoding="utf-8",
    )
    data_config_path.write_text(
        yaml.safe_dump(
            {
                "train_positive_path": str(train_interactions_path),
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
                    "metrics_report_path": str(training_report_path),
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
                    "model_comparison_path": str(model_comparison_path),
                    "selected_model_path": str(selected_model_path),
                }
            }
        ),
        encoding="utf-8",
    )

    monkeypatch.setattr(
        evaluate_pipeline,
        "PARAMS_PATH",
        params_path,
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
    monkeypatch.setattr(
        evaluate_pipeline,
        "_build_models",
        _build_fake_models,
    )
    monkeypatch.setenv(
        "MLFLOW_TRACKING_URI",
        f"sqlite:///{mlflow_database.as_posix()}",
    )
    monkeypatch.setenv(
        "MLFLOW_EXPERIMENT_NAME",
        "evaluation-pipeline-integration-test",
    )

    results = evaluate_pipeline.run_evaluation()

    assert len(results) == EXPECTED_MODEL_COUNT

    for model_name in (
        "popularity",
        "item_knn",
        "neural_cf",
    ):
        report_path = output_directory / f"{model_name}_metrics.json"
        assert report_path.exists()

        report = json.loads(report_path.read_text(encoding="utf-8"))

        assert report["model_name"] == model_name
        assert report["metrics"]["evaluated_users"] == EXPECTED_EVALUATED_USERS
        assert report["metrics"]["k"] == DEFAULT_K

    assert model_comparison_path.exists()

    comparison = pd.read_csv(model_comparison_path)

    assert selected_model_path.exists()

    selected_model = json.loads(selected_model_path.read_text(encoding="utf-8"))

    assert selected_model["model_name"] == "item_knn"
    assert selected_model["primary_metric"] == "ndcg_at_k"
    assert selected_model["k"] == DEFAULT_K
    assert selected_model["evaluated_users"] == EXPECTED_EVALUATED_USERS

    assert set(comparison["model_name"]) == {
        "popularity",
        "item_knn",
        "neural_cf",
    }
