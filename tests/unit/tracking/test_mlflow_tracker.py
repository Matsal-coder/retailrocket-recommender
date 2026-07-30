"""Unit tests for the MLflow tracker."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from retail_recommender.tracking.mlflow_tracker import (
    DEFAULT_EXPERIMENT_NAME,
    MLflowTracker,
)

CUSTOM_EXPERIMENT_NAME = "test-experiment"
CUSTOM_TRACKING_URI = "http://127.0.0.1:5000"
TEST_EPOCH = 2


@patch("retail_recommender.tracking.mlflow_tracker.mlflow")
def test_tracker_configures_explicit_tracking_uri(
    mlflow_mock: MagicMock,
) -> None:
    tracker = MLflowTracker(
        tracking_uri=CUSTOM_TRACKING_URI,
        experiment_name=CUSTOM_EXPERIMENT_NAME,
    )

    mlflow_mock.set_tracking_uri.assert_called_once_with(CUSTOM_TRACKING_URI)
    mlflow_mock.set_experiment.assert_called_once_with(CUSTOM_EXPERIMENT_NAME)
    assert tracker.tracking_uri == CUSTOM_TRACKING_URI


@patch("retail_recommender.tracking.mlflow_tracker.mlflow")
def test_tracker_uses_default_experiment_name(
    mlflow_mock: MagicMock,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv(
        "MLFLOW_EXPERIMENT_NAME",
        raising=False,
    )

    tracker = MLflowTracker()

    mlflow_mock.set_experiment.assert_called_once_with(DEFAULT_EXPERIMENT_NAME)
    assert tracker.experiment_name == DEFAULT_EXPERIMENT_NAME


@patch("retail_recommender.tracking.mlflow_tracker.mlflow")
def test_tracker_uses_environment_configuration(
    mlflow_mock: MagicMock,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv(
        "MLFLOW_TRACKING_URI",
        CUSTOM_TRACKING_URI,
    )
    monkeypatch.setenv(
        "MLFLOW_EXPERIMENT_NAME",
        CUSTOM_EXPERIMENT_NAME,
    )

    tracker = MLflowTracker()

    assert tracker.tracking_uri == CUSTOM_TRACKING_URI
    assert tracker.experiment_name == CUSTOM_EXPERIMENT_NAME
    mlflow_mock.set_tracking_uri.assert_called_once_with(CUSTOM_TRACKING_URI)


@patch("retail_recommender.tracking.mlflow_tracker.mlflow")
def test_tracker_keeps_mlflow_default_when_uri_is_absent(
    mlflow_mock: MagicMock,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv(
        "MLFLOW_TRACKING_URI",
        raising=False,
    )

    tracker = MLflowTracker(experiment_name=CUSTOM_EXPERIMENT_NAME)

    assert tracker.tracking_uri is None
    mlflow_mock.set_tracking_uri.assert_not_called()


def test_tracker_rejects_empty_experiment_name() -> None:
    with pytest.raises(
        ValueError,
        match="experiment_name must not be empty",
    ):
        MLflowTracker(experiment_name=" ")


@patch("retail_recommender.tracking.mlflow_tracker.mlflow")
def test_run_starts_named_mlflow_run(
    mlflow_mock: MagicMock,
) -> None:
    active_run = MagicMock()
    context_manager = MagicMock()
    context_manager.__enter__.return_value = active_run
    mlflow_mock.start_run.return_value = context_manager

    tracker = MLflowTracker(experiment_name=CUSTOM_EXPERIMENT_NAME)

    with tracker.run(
        run_name="popularity",
        tags={"model_type": "baseline"},
    ) as returned_run:
        assert returned_run is active_run

    mlflow_mock.start_run.assert_called_once_with(
        run_name="popularity",
        tags={"model_type": "baseline"},
    )
    context_manager.__exit__.assert_called_once()


@patch("retail_recommender.tracking.mlflow_tracker.mlflow")
def test_log_parameters_normalizes_structured_values(
    mlflow_mock: MagicMock,
    tmp_path: Path,
) -> None:
    MLflowTracker.log_parameters(
        {
            "hidden_layers": [64, 32],
            "checkpoint_path": tmp_path / "model.pt",
            "learning_rate": 0.001,
        }
    )

    mlflow_mock.log_params.assert_called_once_with(
        {
            "hidden_layers": "64,32",
            "checkpoint_path": str(tmp_path / "model.pt"),
            "learning_rate": 0.001,
        }
    )


@patch("retail_recommender.tracking.mlflow_tracker.mlflow")
def test_log_metrics_passes_step(
    mlflow_mock: MagicMock,
) -> None:
    MLflowTracker.log_metrics(
        {
            "train_loss": 0.5,
            "validation_loss": 0.6,
        },
        step=TEST_EPOCH,
    )

    mlflow_mock.log_metrics.assert_called_once_with(
        {
            "train_loss": 0.5,
            "validation_loss": 0.6,
        },
        step=TEST_EPOCH,
    )


@patch("retail_recommender.tracking.mlflow_tracker.mlflow")
def test_log_training_history_uses_epoch_as_step(
    mlflow_mock: MagicMock,
) -> None:
    MLflowTracker.log_training_history(
        [
            {
                "epoch": 1,
                "train_loss": 0.7,
                "validation_loss": 0.8,
                "improved": True,
            },
            {
                "epoch": 2,
                "train_loss": 0.6,
                "validation_loss": 0.65,
                "improved": True,
            },
        ]
    )

    assert mlflow_mock.log_metrics.call_count == TEST_EPOCH

    mlflow_mock.log_metrics.assert_any_call(
        {
            "train_loss": 0.7,
            "validation_loss": 0.8,
        },
        step=1,
    )
    mlflow_mock.log_metrics.assert_any_call(
        {
            "train_loss": 0.6,
            "validation_loss": 0.65,
        },
        step=2,
    )


@patch("retail_recommender.tracking.mlflow_tracker.mlflow")
def test_log_artifact_logs_existing_file(
    mlflow_mock: MagicMock,
    tmp_path: Path,
) -> None:
    artifact_path = tmp_path / "metrics.json"
    artifact_path.write_text(
        "{}",
        encoding="utf-8",
    )

    MLflowTracker.log_artifact(
        artifact_path,
        artifact_directory="reports",
    )

    mlflow_mock.log_artifact.assert_called_once_with(
        str(artifact_path),
        artifact_path="reports",
    )


def test_log_artifact_rejects_missing_file(
    tmp_path: Path,
) -> None:
    artifact_path = tmp_path / "missing.json"

    with pytest.raises(
        FileNotFoundError,
        match="Artifact file not found",
    ):
        MLflowTracker.log_artifact(artifact_path)


def test_log_artifact_rejects_directory(
    tmp_path: Path,
) -> None:
    with pytest.raises(
        ValueError,
        match="Artifact path must be a file",
    ):
        MLflowTracker.log_artifact(tmp_path)
