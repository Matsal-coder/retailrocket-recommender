"""MLflow experiment tracking utilities."""

from __future__ import annotations

from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import mlflow
from mlflow.entities import Run

from retail_recommender.config.settings import get_settings


class MLflowTracker:
    """Provide a small project-specific interface over MLflow Tracking."""

    def __init__(
        self,
        *,
        tracking_uri: str | None = None,
        experiment_name: str | None = None,
    ) -> None:
        """Initialize the MLflow tracker.

        Args:
            tracking_uri: Optional MLflow tracking URI. When omitted, the
                environment variable or MLflow local default is used.
            experiment_name: Experiment name. When omitted, the environment
                variable or project default is used.

        Raises:
            ValueError: If an explicitly supplied experiment name is empty.
        """
        settings = get_settings()
        resolved_tracking_uri = (
            tracking_uri if tracking_uri is not None else settings.mlflow_tracking_uri
        )
        resolved_experiment_name = (
            experiment_name
            if experiment_name is not None
            else settings.mlflow_experiment_name
        )

        if not resolved_experiment_name.strip():
            msg = "experiment_name must not be empty"
            raise ValueError(msg)

        self.tracking_uri = (
            resolved_tracking_uri.strip() if resolved_tracking_uri else None
        )
        self.experiment_name = resolved_experiment_name.strip()

        self.configure()

    def configure(self) -> None:
        """Configure MLflow tracking and select the project experiment."""
        if self.tracking_uri is not None:
            mlflow.set_tracking_uri(self.tracking_uri)

        mlflow.set_experiment(self.experiment_name)

    @contextmanager
    def run(
        self,
        *,
        run_name: str,
        tags: Mapping[str, str] | None = None,
    ) -> Iterator[Run]:
        """Start and safely terminate an MLflow run.

        Args:
            run_name: Human-readable run name.
            tags: Optional run tags.

        Yields:
            Active MLflow run.

        Raises:
            ValueError: If run_name is empty.
        """
        if not run_name.strip():
            msg = "run_name must not be empty"
            raise ValueError(msg)

        with mlflow.start_run(
            run_name=run_name.strip(),
            tags=dict(tags or {}),
        ) as active_run:
            yield active_run

    @staticmethod
    def log_parameters(
        parameters: Mapping[str, Any],
    ) -> None:
        """Log flattened model and training parameters.

        Args:
            parameters: Parameter names and scalar-compatible values.
        """
        if not parameters:
            return

        normalized_parameters = {
            key: MLflowTracker._normalize_parameter_value(value)
            for key, value in parameters.items()
        }

        mlflow.log_params(normalized_parameters)

    @staticmethod
    def log_metrics(
        metrics: Mapping[str, int | float],
        *,
        step: int | None = None,
    ) -> None:
        """Log numerical metrics.

        Args:
            metrics: Metric names and numerical values.
            step: Optional iteration or epoch number.
        """
        if not metrics:
            return

        normalized_metrics = {key: float(value) for key, value in metrics.items()}

        mlflow.log_metrics(
            normalized_metrics,
            step=step,
        )

    @staticmethod
    def log_training_history(
        history: list[Mapping[str, int | float | bool]],
    ) -> None:
        """Log epoch-level training and validation losses.

        Args:
            history: Trainer history containing epoch and loss values.
        """
        for epoch_result in history:
            epoch = int(epoch_result["epoch"])

            MLflowTracker.log_metrics(
                {
                    "train_loss": float(epoch_result["train_loss"]),
                    "validation_loss": float(epoch_result["validation_loss"]),
                },
                step=epoch,
            )

    @staticmethod
    def log_artifact(
        artifact_path: Path,
        *,
        artifact_directory: str | None = None,
    ) -> None:
        """Log one output file as an MLflow artifact.

        Args:
            artifact_path: Existing file to log.
            artifact_directory: Optional destination directory in the run.

        Raises:
            FileNotFoundError: If the artifact does not exist.
            ValueError: If artifact_path is not a file.
        """
        if not artifact_path.exists():
            msg = f"Artifact file not found: {artifact_path}"
            raise FileNotFoundError(msg)

        if not artifact_path.is_file():
            msg = f"Artifact path must be a file: {artifact_path}"
            raise ValueError(msg)

        mlflow.log_artifact(
            str(artifact_path),
            artifact_path=artifact_directory,
        )

    @staticmethod
    def set_tags(tags: Mapping[str, str]) -> None:
        """Set additional tags on the active run."""
        if tags:
            mlflow.set_tags(dict(tags))

    @staticmethod
    def _normalize_parameter_value(value: Any) -> Any:
        """Convert structured parameter values to stable representations."""
        if isinstance(value, list | tuple):
            return ",".join(str(item) for item in value)

        if isinstance(value, Path):
            return str(value)

        return value
