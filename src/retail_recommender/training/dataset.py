"""PyTorch dataset for implicit-feedback recommendation."""

from __future__ import annotations

from typing import TypedDict

import pandas as pd
import torch
from torch import Tensor
from torch.utils.data import Dataset

REQUIRED_COLUMNS = {
    "user_idx",
    "item_idx",
    "target",
}


class InteractionSample(TypedDict):
    """Single implicit-feedback training sample."""

    user_idx: Tensor
    item_idx: Tensor
    target: Tensor


class ImplicitFeedbackDataset(Dataset[InteractionSample]):
    """Store encoded user-item interactions as PyTorch tensors."""

    def __init__(self, interactions: pd.DataFrame) -> None:
        """Initialize the dataset from processed interactions.

        Args:
            interactions: DataFrame containing encoded users, encoded items
                and binary targets.

        Raises:
            TypeError: If interactions is not a pandas DataFrame.
            ValueError: If the DataFrame is empty, has missing columns,
                contains null values, invalid indices or non-binary targets.
        """
        if not isinstance(interactions, pd.DataFrame):
            msg = "interactions must be a pandas DataFrame"
            raise TypeError(msg)

        self._validate_columns(interactions)

        if interactions.empty:
            msg = "interactions must not be empty"
            raise ValueError(msg)

        relevant_data = interactions[["user_idx", "item_idx", "target"]].copy()

        self._validate_null_values(relevant_data)
        self._validate_indices(relevant_data)
        self._validate_targets(relevant_data)

        self._user_indices = torch.tensor(
            relevant_data["user_idx"].to_numpy(),
            dtype=torch.long,
        )
        self._item_indices = torch.tensor(
            relevant_data["item_idx"].to_numpy(),
            dtype=torch.long,
        )
        self._targets = torch.tensor(
            relevant_data["target"].to_numpy(),
            dtype=torch.float32,
        )

    def __len__(self) -> int:
        """Return the number of user-item samples."""
        return len(self._targets)

    def __getitem__(self, index: int) -> InteractionSample:
        """Return one user-item-target sample.

        Args:
            index: Positional sample index.

        Returns:
            Dictionary containing user, item and target tensors.
        """
        return {
            "user_idx": self._user_indices[index],
            "item_idx": self._item_indices[index],
            "target": self._targets[index],
        }

    @property
    def user_indices(self) -> Tensor:
        """Return a copy of all encoded user indices."""
        return self._user_indices.clone()

    @property
    def item_indices(self) -> Tensor:
        """Return a copy of all encoded item indices."""
        return self._item_indices.clone()

    @property
    def targets(self) -> Tensor:
        """Return a copy of all binary targets."""
        return self._targets.clone()

    @staticmethod
    def _validate_columns(interactions: pd.DataFrame) -> None:
        """Validate required DataFrame columns."""
        missing_columns = REQUIRED_COLUMNS.difference(interactions.columns)

        if missing_columns:
            formatted_columns = ", ".join(sorted(missing_columns))
            msg = "Missing required interaction columns: " f"{formatted_columns}"
            raise ValueError(msg)

    @staticmethod
    def _validate_null_values(interactions: pd.DataFrame) -> None:
        """Reject null values in required columns."""
        columns_with_nulls = interactions.columns[interactions.isna().any()].tolist()

        if columns_with_nulls:
            formatted_columns = ", ".join(columns_with_nulls)
            msg = (
                "Interaction columns must not contain null values: "
                f"{formatted_columns}"
            )
            raise ValueError(msg)

    @staticmethod
    def _validate_indices(interactions: pd.DataFrame) -> None:
        """Validate that user and item indices are non-negative integers."""
        for column_name in ("user_idx", "item_idx"):
            values = interactions[column_name]

            if not pd.api.types.is_integer_dtype(values):
                msg = f"{column_name} must contain integer values"
                raise ValueError(msg)

            if (values < 0).any():
                msg = f"{column_name} must contain non-negative values"
                raise ValueError(msg)

    @staticmethod
    def _validate_targets(interactions: pd.DataFrame) -> None:
        """Validate binary implicit-feedback targets."""
        target_values = set(interactions["target"].astype(float).unique().tolist())

        if not target_values.issubset({0.0, 1.0}):
            msg = "target must contain only binary values: 0 or 1"
            raise ValueError(msg)
