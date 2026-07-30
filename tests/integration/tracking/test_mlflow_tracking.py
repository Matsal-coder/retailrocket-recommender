"""Integration test for local MLflow experiment tracking."""

from __future__ import annotations

from pathlib import Path

from mlflow.tracking import MlflowClient

from retail_recommender.tracking.mlflow_tracker import (
    MLflowTracker,
)

EXPERIMENT_NAME = "integration-test-experiment"
RUN_NAME = "integration-test-run"

DEFAULT_NDGC = 0.4


def test_mlflow_tracker_creates_local_run(
    tmp_path: Path,
) -> None:
    tracking_database = tmp_path / "mlflow.db"
    tracking_uri = f"sqlite:///{tracking_database.as_posix()}"

    artifact_path = tmp_path / "metrics.json"
    artifact_path.write_text(
        '{"ndcg_at_10": 0.4}',
        encoding="utf-8",
    )

    tracker = MLflowTracker(
        tracking_uri=tracking_uri,
        experiment_name=EXPERIMENT_NAME,
    )

    with tracker.run(
        run_name=RUN_NAME,
        tags={
            "model_type": "integration-test",
        },
    ) as active_run:
        tracker.log_parameters(
            {
                "embedding_dim": 4,
                "hidden_layers": [8, 4],
            }
        )
        tracker.log_metrics(
            {
                "validation_loss": 0.5,
                "ndcg_at_10": 0.4,
            }
        )
        tracker.log_artifact(
            artifact_path,
            artifact_directory="reports",
        )

        run_id = active_run.info.run_id

    client = MlflowClient(tracking_uri=tracking_uri)
    stored_run = client.get_run(run_id)

    assert stored_run.data.params["embedding_dim"] == "4"
    assert stored_run.data.params["hidden_layers"] == "8,4"
    assert stored_run.data.metrics["ndcg_at_10"] == DEFAULT_NDGC
    assert stored_run.data.tags["mlflow.runName"] == RUN_NAME

    artifacts = client.list_artifacts(
        run_id,
        path="reports",
    )

    assert len(artifacts) == 1
    assert artifacts[0].path == "reports/metrics.json"
