"""Integration test for the training pipeline."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import torch
import yaml

from retail_recommender.pipelines import train as train_pipeline

TEST_SEED = 731
EXPECTED_HISTORY_LENGTH = 2


def test_train_pipeline_generates_outputs(
    tmp_path: Path,
    monkeypatch,
) -> None:
    train_path = tmp_path / "train.parquet"
    validation_path = tmp_path / "validation.parquet"
    checkpoint_path = tmp_path / "models" / "best_model.pt"
    report_path = tmp_path / "reports" / "train_metrics.json"
    mlflow_database = tmp_path / "mlflow.db"

    train_interactions = pd.DataFrame(
        {
            "user_idx": [0, 0, 1, 1, 2, 2],
            "item_idx": [0, 1, 1, 2, 2, 3],
            "target": [1, 0, 1, 0, 1, 0],
        }
    )
    validation_interactions = pd.DataFrame(
        {
            "user_idx": [0, 1, 2],
            "item_idx": [0, 1, 2],
            "target": [1, 1, 1],
        }
    )

    train_interactions.to_parquet(
        train_path,
        index=False,
    )
    validation_interactions.to_parquet(
        validation_path,
        index=False,
    )

    params_path = tmp_path / "params.yaml"
    model_config_path = tmp_path / "model.yaml"
    training_config_path = tmp_path / "training.yaml"

    params_path.write_text(
        yaml.safe_dump(
            {
                "model": {
                    "name": "neural_cf",
                    "embedding_dim": 4,
                    "hidden_layers": [8, 4],
                    "dropout": 0.0,
                },
                "training": {
                    "random_seed": TEST_SEED,
                    "batch_size": 2,
                    "learning_rate": 0.01,
                    "epochs": EXPECTED_HISTORY_LENGTH,
                    "patience": EXPECTED_HISTORY_LENGTH,
                    "minimum_delta": 0.0,
                    "weight_decay": 0.0,
                    "device": "cpu",
                },
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
                    "train_data_path": str(train_path),
                    "validation_data_path": str(validation_path),
                    "metrics_report_path": str(report_path),
                }
            }
        ),
        encoding="utf-8",
    )

    monkeypatch.setattr(
        train_pipeline,
        "PARAMS_PATH",
        params_path,
    )
    monkeypatch.setattr(
        train_pipeline,
        "MODEL_CONFIG_PATH",
        model_config_path,
    )
    monkeypatch.setattr(
        train_pipeline,
        "TRAINING_CONFIG_PATH",
        training_config_path,
    )
    monkeypatch.setenv(
        "MLFLOW_TRACKING_URI",
        f"sqlite:///{mlflow_database.as_posix()}",
    )
    monkeypatch.setenv(
        "MLFLOW_EXPERIMENT_NAME",
        "train-pipeline-integration-test",
    )

    result = train_pipeline.run_training()

    assert checkpoint_path.exists()
    assert report_path.exists()
    assert result.completed_epochs >= 1

    state_dict = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=True,
    )
    assert state_dict

    report = json.loads(report_path.read_text(encoding="utf-8"))

    assert report["model"]["model_name"] == "neural_cf"
    assert report["model"]["random_seed"] == TEST_SEED
    assert len(report["history"]) == result.completed_epochs
