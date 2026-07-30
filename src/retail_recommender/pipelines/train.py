"""Train the Neural Collaborative Filtering model."""

from __future__ import annotations

import json
import logging
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import pandas as pd
import torch
import yaml
from torch.utils.data import DataLoader

from retail_recommender.models.neural_cf import (
    NeuralCollaborativeFiltering,
)
from retail_recommender.tracking.mlflow_tracker import MLflowTracker
from retail_recommender.training.dataset import (
    ImplicitFeedbackDataset,
)
from retail_recommender.training.seed import set_global_seed
from retail_recommender.training.trainer import (
    Trainer,
    TrainingResult,
)

LOGGER = logging.getLogger(__name__)

PARAMS_PATH = Path("params.yaml")
MODEL_CONFIG_PATH = Path("configs/model.yaml")
TRAINING_CONFIG_PATH = Path("configs/training.yaml")

REQUIRED_INTERACTION_COLUMNS = {
    "user_idx",
    "item_idx",
    "target",
}


def run_training() -> TrainingResult:
    """Run the complete neural recommendation training pipeline."""
    params = _load_yaml(PARAMS_PATH)
    model_config = _load_yaml(MODEL_CONFIG_PATH)
    training_config = _load_yaml(TRAINING_CONFIG_PATH)

    model_params = _get_mapping(params, "model")
    training_params = _get_mapping(params, "training")

    model_paths = _get_mapping(model_config, "model")
    training_paths = _get_mapping(
        training_config,
        "training",
    )

    seed = _get_required_int(
        training_params,
        "random_seed",
    )
    set_global_seed(seed)

    train_path = Path(
        _get_required_string(
            training_paths,
            "train_data_path",
        )
    )
    validation_path = Path(
        _get_required_string(
            training_paths,
            "validation_data_path",
        )
    )
    checkpoint_path = Path(
        _get_required_string(
            model_paths,
            "checkpoint_path",
        )
    )
    metrics_report_path = Path(
        _get_required_string(
            training_paths,
            "metrics_report_path",
        )
    )

    LOGGER.info("Loading training data from %s", train_path)
    train_interactions = pd.read_parquet(train_path)

    LOGGER.info(
        "Loading validation data from %s",
        validation_path,
    )
    validation_interactions = pd.read_parquet(validation_path)

    _validate_interactions(
        train_interactions,
        split_name="training",
    )
    _validate_interactions(
        validation_interactions,
        split_name="validation",
    )

    num_users, num_items = _resolve_entity_counts(
        train_interactions,
        validation_interactions,
    )

    batch_size = _get_required_int(
        training_params,
        "batch_size",
    )

    train_loader = _create_data_loader(
        train_interactions,
        batch_size=batch_size,
        shuffle=True,
        seed=seed,
    )
    validation_loader = _create_data_loader(
        validation_interactions,
        batch_size=batch_size,
        shuffle=False,
        seed=seed,
    )

    embedding_dim = _get_required_int(
        model_params,
        "embedding_dim",
    )
    hidden_layers = _get_required_int_sequence(
        model_params,
        "hidden_layers",
    )
    dropout = _get_required_float(
        model_params,
        "dropout",
    )

    model = NeuralCollaborativeFiltering(
        num_users=num_users,
        num_items=num_items,
        embedding_dim=embedding_dim,
        hidden_layers=hidden_layers,
        dropout=dropout,
    )

    device = _resolve_device(
        _get_required_string(
            training_params,
            "device",
        )
    )

    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        validation_loader=validation_loader,
        learning_rate=_get_required_float(
            training_params,
            "learning_rate",
        ),
        epochs=_get_required_int(
            training_params,
            "epochs",
        ),
        patience=_get_required_int(
            training_params,
            "patience",
        ),
        minimum_delta=_get_required_float(
            training_params,
            "minimum_delta",
        ),
        weight_decay=_get_required_float(
            training_params,
            "weight_decay",
        ),
        checkpoint_path=checkpoint_path,
        device=device,
    )

    tracker = MLflowTracker()

    with tracker.run(
        run_name="neural_cf_train",
        tags={
            "model_type": "neural",
            "pipeline": "train",
        },
    ):
        tracker.log_parameters(
            {
                "random_seed": seed,
                "num_users": num_users,
                "num_items": num_items,
                "embedding_dim": embedding_dim,
                "hidden_layers": hidden_layers,
                "dropout": dropout,
                "batch_size": batch_size,
                "learning_rate": training_params["learning_rate"],
                "epochs": training_params["epochs"],
                "patience": training_params["patience"],
                "minimum_delta": training_params["minimum_delta"],
                "weight_decay": training_params["weight_decay"],
                "device": str(device),
            }
        )

        training_result = trainer.fit()

        tracker.log_training_history(training_result.history)
        tracker.log_metrics(
            {
                "best_validation_loss": (training_result.best_validation_loss),
                "best_epoch": training_result.best_epoch,
                "completed_epochs": (training_result.completed_epochs),
            }
        )

        _write_training_report(
            result=training_result,
            path=metrics_report_path,
            model_metadata={
                "model_name": "neural_cf",
                "num_users": num_users,
                "num_items": num_items,
                "embedding_dim": embedding_dim,
                "hidden_layers": list(hidden_layers),
                "dropout": dropout,
                "random_seed": seed,
                "device": str(device),
            },
        )

        tracker.log_artifact(
            checkpoint_path,
            artifact_directory="models",
        )
        tracker.log_artifact(
            metrics_report_path,
            artifact_directory="reports",
        )

    LOGGER.info(
        "Training completed. Best validation loss: %.6f",
        training_result.best_validation_loss,
    )

    return training_result


def _create_data_loader(
    interactions: pd.DataFrame,
    *,
    batch_size: int,
    shuffle: bool,
    seed: int,
) -> DataLoader:
    """Create a deterministic PyTorch DataLoader."""
    generator = torch.Generator()
    generator.manual_seed(seed)

    dataset = ImplicitFeedbackDataset(interactions)

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        generator=generator,
        num_workers=0,
    )


def _resolve_entity_counts(
    train_interactions: pd.DataFrame,
    validation_interactions: pd.DataFrame,
) -> tuple[int, int]:
    """Resolve embedding table sizes from encoded IDs."""
    combined_users = pd.concat(
        [
            train_interactions["user_idx"],
            validation_interactions["user_idx"],
        ],
        ignore_index=True,
    )
    combined_items = pd.concat(
        [
            train_interactions["item_idx"],
            validation_interactions["item_idx"],
        ],
        ignore_index=True,
    )

    num_users = int(combined_users.max()) + 1
    num_items = int(combined_items.max()) + 1

    return num_users, num_items


def _validate_interactions(
    interactions: pd.DataFrame,
    *,
    split_name: str,
) -> None:
    """Validate an interaction split before training."""
    missing_columns = REQUIRED_INTERACTION_COLUMNS.difference(interactions.columns)

    if missing_columns:
        formatted_columns = ", ".join(sorted(missing_columns))
        msg = f"{split_name} data is missing columns: " f"{formatted_columns}"
        raise ValueError(msg)

    if interactions.empty:
        msg = f"{split_name} data must not be empty"
        raise ValueError(msg)


def _resolve_device(device_name: str) -> torch.device:
    """Resolve configured training device."""
    normalized_name = device_name.strip().lower()

    if normalized_name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if normalized_name == "cuda":
        if not torch.cuda.is_available():
            msg = "CUDA was requested but is not available"
            raise RuntimeError(msg)

        return torch.device("cuda")

    if normalized_name == "cpu":
        return torch.device("cpu")

    msg = "device must be one of: auto, cpu, cuda"
    raise ValueError(msg)


def _write_training_report(
    *,
    result: TrainingResult,
    path: Path,
    model_metadata: Mapping[str, Any],
) -> None:
    """Write training metrics and metadata as JSON."""
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    payload = {
        "best_epoch": result.best_epoch,
        "best_validation_loss": (result.best_validation_loss),
        "completed_epochs": result.completed_epochs,
        "stopped_early": result.stopped_early,
        "checkpoint_path": result.checkpoint_path,
        "history": result.history,
        "model": dict(model_metadata),
    }

    path.write_text(
        json.dumps(
            payload,
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )


def _load_yaml(path: Path) -> dict[str, Any]:
    """Load a YAML mapping from disk."""
    if not path.exists():
        msg = f"Configuration file not found: {path}"
        raise FileNotFoundError(msg)

    loaded_content = yaml.safe_load(path.read_text(encoding="utf-8"))

    if not isinstance(loaded_content, dict):
        msg = f"Configuration must be a mapping: {path}"
        raise ValueError(msg)

    return loaded_content


def _get_mapping(
    values: Mapping[str, Any],
    key: str,
) -> Mapping[str, Any]:
    """Read a required nested configuration mapping."""
    nested_values = values.get(key)

    if not isinstance(nested_values, dict):
        msg = f"Missing configuration section: {key}"
        raise ValueError(msg)

    return nested_values


def _get_required_string(
    values: Mapping[str, Any],
    key: str,
) -> str:
    """Read a required non-empty string."""
    value = values.get(key)

    if not isinstance(value, str) or not value.strip():
        msg = f"Configuration value must be a string: {key}"
        raise ValueError(msg)

    return value.strip()


def _get_required_int(
    values: Mapping[str, Any],
    key: str,
) -> int:
    """Read a required integer."""
    value = values.get(key)

    if not isinstance(value, int) or isinstance(value, bool):
        msg = f"Configuration value must be an integer: {key}"
        raise ValueError(msg)

    return value


def _get_required_float(
    values: Mapping[str, Any],
    key: str,
) -> float:
    """Read a required numerical value as float."""
    value = values.get(key)

    if not isinstance(value, int | float) or isinstance(value, bool):
        msg = f"Configuration value must be numeric: {key}"
        raise ValueError(msg)

    return float(value)


def _get_required_int_sequence(
    values: Mapping[str, Any],
    key: str,
) -> tuple[int, ...]:
    """Read a required sequence of integer values."""
    value = values.get(key)

    if not isinstance(value, list | tuple) or not value:
        msg = "Configuration value must be a non-empty " f"integer sequence: {key}"
        raise ValueError(msg)

    if any(not isinstance(item, int) or isinstance(item, bool) for item in value):
        msg = "Configuration value must contain integers: " f"{key}"
        raise ValueError(msg)

    return tuple(value)


def main() -> None:
    """Execute the training pipeline."""
    run_training()


if __name__ == "__main__":
    main()
