"""Early stopping based on validation loss."""

from __future__ import annotations

import math


class EarlyStopping:
    """Stop training after validation loss stops improving."""

    def __init__(
        self,
        patience: int,
        minimum_delta: float = 0.0,
    ) -> None:
        """Initialize early stopping state.

        Args:
            patience: Number of consecutive non-improving epochs allowed.
            minimum_delta: Minimum decrease required to count as improvement.

        Raises:
            ValueError: If configuration values are invalid.
        """
        if patience <= 0:
            msg = "patience must be greater than zero"
            raise ValueError(msg)

        if minimum_delta < 0.0:
            msg = "minimum_delta must be non-negative"
            raise ValueError(msg)

        self.patience = patience
        self.minimum_delta = minimum_delta

        self.best_loss = math.inf
        self.bad_epoch_count = 0
        self.should_stop = False

    def update(self, validation_loss: float) -> bool:
        """Update the state with the latest validation loss.

        Args:
            validation_loss: Mean validation loss for the current epoch.

        Returns:
            True when the loss improved enough to reset patience.

        Raises:
            ValueError: If validation_loss is not finite.
        """
        if not math.isfinite(validation_loss):
            msg = "validation_loss must be finite"
            raise ValueError(msg)

        improved = validation_loss < self.best_loss - self.minimum_delta

        if improved:
            self.best_loss = validation_loss
            self.bad_epoch_count = 0
            self.should_stop = False
            return True

        self.bad_epoch_count += 1
        self.should_stop = self.bad_epoch_count >= self.patience

        return False
