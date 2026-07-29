"""Unit tests for the neural model Trainer."""

from pathlib import Path
from typing import Any

import pandas as pd
import pytest
import torch
from torch.utils.data import DataLoader

from retail_recommender.models.neural_cf import (
    NeuralCollaborativeFiltering,
)
from retail_recommender.training.dataset import (
    ImplicitFeedbackDataset,
)
from retail_recommender.training.trainer import Trainer

NUM_USERS = 3
NUM_ITEMS = 4
BATCH_SIZE = 2
LEARNING_RATE = 0.01
MAX_EPOCHS = 2
PATIENCE = 2


def _create_loader() -> DataLoader:
    interactions = pd.DataFrame(
        {
            "user_idx": [0, 0, 1, 1, 2, 2],
            "item_idx": [0, 1, 1, 2, 2, 3],
            "target": [1, 0, 1, 0, 1, 0],
        }
    )
    dataset = ImplicitFeedbackDataset(interactions)

    return DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
    )


def _create_model() -> NeuralCollaborativeFiltering:
    return NeuralCollaborativeFiltering(
        num_users=NUM_USERS,
        num_items=NUM_ITEMS,
        embedding_dim=4,
        hidden_layers=(8, 4),
        dropout=0.0,
    )


def test_trainer_saves_checkpoint(
    tmp_path: Path,
) -> None:
    checkpoint_path = tmp_path / "best_model.pt"
    loader = _create_loader()
    trainer = Trainer(
        model=_create_model(),
        train_loader=loader,
        validation_loader=loader,
        learning_rate=LEARNING_RATE,
        epochs=MAX_EPOCHS,
        patience=PATIENCE,
        checkpoint_path=checkpoint_path,
        device=torch.device("cpu"),
    )

    result = trainer.fit()

    assert checkpoint_path.exists()
    assert result.best_epoch >= 1
    assert result.completed_epochs <= MAX_EPOCHS
    assert result.best_validation_loss >= 0.0
    assert len(result.history) == result.completed_epochs


def test_trainer_history_contains_losses(
    tmp_path: Path,
) -> None:
    loader = _create_loader()
    trainer = Trainer(
        model=_create_model(),
        train_loader=loader,
        validation_loader=loader,
        learning_rate=LEARNING_RATE,
        epochs=1,
        patience=PATIENCE,
        checkpoint_path=tmp_path / "model.pt",
        device=torch.device("cpu"),
    )

    result = trainer.fit()
    first_epoch = result.history[0]

    assert "train_loss" in first_epoch
    assert "validation_loss" in first_epoch
    assert "improved" in first_epoch


@pytest.mark.parametrize(
    ("parameter_name", "parameter_value", "expected_message"),
    [
        (
            "learning_rate",
            0.0,
            "learning_rate must be greater than zero",
        ),
        (
            "epochs",
            0,
            "epochs must be greater than zero",
        ),
        (
            "weight_decay",
            -0.1,
            "weight_decay must be non-negative",
        ),
    ],
)
def test_trainer_rejects_invalid_configuration(
    parameter_name: str,
    parameter_value: int | float,
    expected_message: str,
    tmp_path: Path,
) -> None:
    loader = _create_loader()
    parameters: dict[str, Any] = {
        "model": _create_model(),
        "train_loader": loader,
        "validation_loader": loader,
        "learning_rate": LEARNING_RATE,
        "epochs": MAX_EPOCHS,
        "patience": PATIENCE,
        "checkpoint_path": tmp_path / "model.pt",
        "device": torch.device("cpu"),
        "weight_decay": 0.0,
    }
    parameters[parameter_name] = parameter_value

    with pytest.raises(ValueError, match=expected_message):
        Trainer(**parameters)
