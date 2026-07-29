"""Integration test for the complete neural training loop."""

from pathlib import Path

import pandas as pd
import torch
from torch.utils.data import DataLoader

from retail_recommender.models.neural_cf import (
    NeuralCollaborativeFiltering,
)
from retail_recommender.training.dataset import (
    ImplicitFeedbackDataset,
)
from retail_recommender.training.seed import set_global_seed
from retail_recommender.training.trainer import Trainer

TEST_SEED = 731
BATCH_SIZE = 4


def test_training_loop_generates_loadable_checkpoint(
    tmp_path: Path,
) -> None:
    set_global_seed(TEST_SEED)

    train_interactions = pd.DataFrame(
        {
            "user_idx": [0, 0, 1, 1, 2, 2, 3, 3],
            "item_idx": [0, 1, 1, 2, 2, 3, 3, 0],
            "target": [1, 0, 1, 0, 1, 0, 1, 0],
        }
    )
    validation_interactions = pd.DataFrame(
        {
            "user_idx": [0, 1, 2, 3],
            "item_idx": [0, 1, 2, 3],
            "target": [1, 1, 1, 1],
        }
    )

    train_loader = DataLoader(
        ImplicitFeedbackDataset(train_interactions),
        batch_size=BATCH_SIZE,
        shuffle=True,
    )
    validation_loader = DataLoader(
        ImplicitFeedbackDataset(validation_interactions),
        batch_size=BATCH_SIZE,
        shuffle=False,
    )

    model = NeuralCollaborativeFiltering(
        num_users=4,
        num_items=4,
        embedding_dim=4,
        hidden_layers=(8, 4),
        dropout=0.0,
    )
    checkpoint_path = tmp_path / "best_model.pt"

    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        validation_loader=validation_loader,
        learning_rate=0.01,
        epochs=3,
        patience=2,
        checkpoint_path=checkpoint_path,
        device=torch.device("cpu"),
    )

    result = trainer.fit()

    restored_model = NeuralCollaborativeFiltering(
        num_users=4,
        num_items=4,
        embedding_dim=4,
        hidden_layers=(8, 4),
        dropout=0.0,
    )
    restored_model.load_state_dict(
        torch.load(
            checkpoint_path,
            map_location="cpu",
            weights_only=True,
        )
    )
    restored_model.eval()

    users = torch.tensor([0, 1], dtype=torch.long)
    items = torch.tensor([0, 1], dtype=torch.long)

    with torch.no_grad():
        logits = restored_model(users, items)

    assert result.completed_epochs >= 1
    assert checkpoint_path.exists()
    assert logits.shape == (2,)
    assert torch.isfinite(logits).all()
