"""Training loop for neural recommendation models."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
from torch import Tensor, nn
from torch.optim import Adam
from torch.utils.data import DataLoader

from retail_recommender.training.early_stopping import EarlyStopping

Batch = Mapping[str, Tensor]


@dataclass(frozen=True)
class EpochMetrics:
    """Training and validation metrics for one epoch."""

    epoch: int
    train_loss: float
    validation_loss: float
    improved: bool


@dataclass(frozen=True)
class TrainingResult:
    """Summary returned after model training."""

    best_epoch: int
    best_validation_loss: float
    completed_epochs: int
    stopped_early: bool
    checkpoint_path: str
    history: list[dict[str, int | float | bool]]


class Trainer:
    """Train a PyTorch model with validation and early stopping."""

    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader[Batch],
        validation_loader: DataLoader[Batch],
        *,
        learning_rate: float,
        epochs: int,
        patience: int,
        checkpoint_path: Path,
        device: torch.device,
        minimum_delta: float = 0.0,
        weight_decay: float = 0.0,
    ) -> None:
        """Initialize the neural model trainer.

        Args:
            model: PyTorch model that returns one logit per user-item pair.
            train_loader: Training batches.
            validation_loader: Validation batches.
            learning_rate: Adam learning rate.
            epochs: Maximum number of epochs.
            patience: Early-stopping patience.
            checkpoint_path: Path used to save the best state dictionary.
            device: Device used during training.
            minimum_delta: Minimum validation improvement.
            weight_decay: Adam L2 regularization strength.

        Raises:
            ValueError: If numeric configuration is invalid.
        """
        self._validate_configuration(
            learning_rate=learning_rate,
            epochs=epochs,
            weight_decay=weight_decay,
        )

        self.model = model.to(device)
        self.train_loader = train_loader
        self.validation_loader = validation_loader
        self.epochs = epochs
        self.checkpoint_path = checkpoint_path
        self.device = device

        self.criterion = nn.BCEWithLogitsLoss()
        self.optimizer = Adam(
            self.model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay,
        )
        self.early_stopping = EarlyStopping(
            patience=patience,
            minimum_delta=minimum_delta,
        )

    def fit(self) -> TrainingResult:
        """Run training and restore the best saved checkpoint."""
        history: list[EpochMetrics] = []
        best_epoch = 0

        for epoch in range(1, self.epochs + 1):
            train_loss = self._train_epoch()
            validation_loss = self._validate_epoch()

            improved = self.early_stopping.update(validation_loss)

            if improved:
                best_epoch = epoch
                self._save_checkpoint()

            history.append(
                EpochMetrics(
                    epoch=epoch,
                    train_loss=train_loss,
                    validation_loss=validation_loss,
                    improved=improved,
                )
            )

            if self.early_stopping.should_stop:
                break

        if best_epoch == 0:
            msg = "Training finished without a valid checkpoint"
            raise RuntimeError(msg)

        self._load_best_checkpoint()

        serialized_history = [asdict(epoch_metrics) for epoch_metrics in history]

        return TrainingResult(
            best_epoch=best_epoch,
            best_validation_loss=self.early_stopping.best_loss,
            completed_epochs=len(history),
            stopped_early=len(history) < self.epochs,
            checkpoint_path=str(self.checkpoint_path),
            history=serialized_history,
        )

    def _train_epoch(self) -> float:
        """Train the model for one complete epoch."""
        self.model.train()

        total_loss = 0.0
        total_samples = 0

        for batch in self.train_loader:
            user_indices, item_indices, targets = self._prepare_batch(batch)

            self.optimizer.zero_grad()

            logits = self.model(
                user_indices,
                item_indices,
            )
            loss = self.criterion(logits, targets)

            loss.backward()
            self.optimizer.step()

            batch_size = targets.shape[0]
            total_loss += loss.item() * batch_size
            total_samples += batch_size

        return self._calculate_mean_loss(
            total_loss=total_loss,
            total_samples=total_samples,
            split_name="training",
        )

    def _validate_epoch(self) -> float:
        """Evaluate validation loss for one epoch."""
        self.model.eval()

        total_loss = 0.0
        total_samples = 0

        with torch.no_grad():
            for batch in self.validation_loader:
                user_indices, item_indices, targets = self._prepare_batch(batch)

                logits = self.model(
                    user_indices,
                    item_indices,
                )
                loss = self.criterion(logits, targets)

                batch_size = targets.shape[0]
                total_loss += loss.item() * batch_size
                total_samples += batch_size

        return self._calculate_mean_loss(
            total_loss=total_loss,
            total_samples=total_samples,
            split_name="validation",
        )

    def _prepare_batch(
        self,
        batch: Batch,
    ) -> tuple[Tensor, Tensor, Tensor]:
        """Move a batch to the configured device."""
        required_keys = {
            "user_idx",
            "item_idx",
            "target",
        }
        missing_keys = required_keys.difference(batch)

        if missing_keys:
            formatted_keys = ", ".join(sorted(missing_keys))
            msg = f"Batch is missing required keys: {formatted_keys}"
            raise ValueError(msg)

        user_indices = batch["user_idx"].to(self.device)
        item_indices = batch["item_idx"].to(self.device)
        targets = batch["target"].to(
            self.device,
            dtype=torch.float32,
        )

        return user_indices, item_indices, targets

    def _save_checkpoint(self) -> None:
        """Save the current model state as the best checkpoint."""
        self.checkpoint_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        torch.save(
            self.model.state_dict(),
            self.checkpoint_path,
        )

    def _load_best_checkpoint(self) -> None:
        """Restore the best saved state into the model."""
        state_dict = torch.load(
            self.checkpoint_path,
            map_location=self.device,
            weights_only=True,
        )
        self.model.load_state_dict(state_dict)

    @staticmethod
    def _calculate_mean_loss(
        *,
        total_loss: float,
        total_samples: int,
        split_name: str,
    ) -> float:
        """Calculate sample-weighted mean loss."""
        if total_samples == 0:
            msg = f"{split_name} loader must contain samples"
            raise ValueError(msg)

        return total_loss / total_samples

    @staticmethod
    def _validate_configuration(
        *,
        learning_rate: float,
        epochs: int,
        weight_decay: float,
    ) -> None:
        """Validate Trainer configuration."""
        if learning_rate <= 0.0:
            msg = "learning_rate must be greater than zero"
            raise ValueError(msg)

        if epochs <= 0:
            msg = "epochs must be greater than zero"
            raise ValueError(msg)

        if weight_decay < 0.0:
            msg = "weight_decay must be non-negative"
            raise ValueError(msg)
