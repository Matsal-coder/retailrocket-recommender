"""Unit tests for the implicit-feedback PyTorch dataset."""

from __future__ import annotations

from typing import Any

import pandas as pd
import pytest
import torch
from torch.utils.data import DataLoader

from retail_recommender.training.dataset import (
    ImplicitFeedbackDataset,
)

EXPECTED_DATASET_LENGTH = 3
EXPECTED_BATCH_SIZE = 2
SECOND_USER_INDEX = 1
SECOND_ITEM_INDEX = 20


@pytest.fixture
def interactions() -> pd.DataFrame:
    """Create valid encoded implicit-feedback interactions."""
    return pd.DataFrame(
        {
            "user_idx": [0, 1, 2],
            "item_idx": [10, 20, 30],
            "target": [1, 0, 1],
            "interaction_score": [5.0, 0.0, 3.0],
        }
    )


def test_dataset_stores_expected_number_of_samples(
    interactions: pd.DataFrame,
) -> None:
    dataset = ImplicitFeedbackDataset(interactions)

    assert len(dataset) == EXPECTED_DATASET_LENGTH


def test_dataset_returns_expected_sample(
    interactions: pd.DataFrame,
) -> None:
    dataset = ImplicitFeedbackDataset(interactions)

    sample = dataset[1]

    assert sample["user_idx"].item() == SECOND_USER_INDEX
    assert sample["item_idx"].item() == SECOND_ITEM_INDEX
    assert sample["target"].item() == 0.0


def test_dataset_uses_expected_tensor_dtypes(
    interactions: pd.DataFrame,
) -> None:
    dataset = ImplicitFeedbackDataset(interactions)
    sample = dataset[0]

    assert sample["user_idx"].dtype == torch.long
    assert sample["item_idx"].dtype == torch.long
    assert sample["target"].dtype == torch.float32


def test_dataset_properties_return_all_tensors(
    interactions: pd.DataFrame,
) -> None:
    dataset = ImplicitFeedbackDataset(interactions)

    assert dataset.user_indices.tolist() == [0, 1, 2]
    assert dataset.item_indices.tolist() == [10, 20, 30]
    assert dataset.targets.tolist() == [1.0, 0.0, 1.0]


def test_dataset_properties_return_tensor_copies(
    interactions: pd.DataFrame,
) -> None:
    dataset = ImplicitFeedbackDataset(interactions)

    user_indices = dataset.user_indices
    user_indices[0] = 999

    assert dataset.user_indices[0].item() == 0


def test_dataset_works_with_pytorch_dataloader(
    interactions: pd.DataFrame,
) -> None:
    dataset = ImplicitFeedbackDataset(interactions)
    data_loader = DataLoader(
        dataset,
        batch_size=EXPECTED_BATCH_SIZE,
        shuffle=False,
    )

    batch = next(iter(data_loader))

    assert batch["user_idx"].shape == (EXPECTED_BATCH_SIZE,)
    assert batch["item_idx"].shape == (EXPECTED_BATCH_SIZE,)
    assert batch["target"].shape == (EXPECTED_BATCH_SIZE,)
    assert batch["user_idx"].dtype == torch.long
    assert batch["target"].dtype == torch.float32


def test_dataset_ignores_unrelated_columns(
    interactions: pd.DataFrame,
) -> None:
    dataset = ImplicitFeedbackDataset(interactions)

    assert len(dataset) == EXPECTED_DATASET_LENGTH


def test_dataset_rejects_non_dataframe() -> None:
    invalid_interactions: Any = [
        {
            "user_idx": 0,
            "item_idx": 10,
            "target": 1,
        }
    ]

    with pytest.raises(
        TypeError,
        match="interactions must be a pandas DataFrame",
    ):
        ImplicitFeedbackDataset(invalid_interactions)


def test_dataset_rejects_empty_dataframe() -> None:
    interactions = pd.DataFrame(
        columns=[
            "user_idx",
            "item_idx",
            "target",
        ]
    )

    with pytest.raises(
        ValueError,
        match="interactions must not be empty",
    ):
        ImplicitFeedbackDataset(interactions)


def test_dataset_rejects_missing_columns() -> None:
    interactions = pd.DataFrame(
        {
            "user_idx": [0],
            "item_idx": [10],
        }
    )

    with pytest.raises(
        ValueError,
        match="target",
    ):
        ImplicitFeedbackDataset(interactions)


@pytest.mark.parametrize(
    "column_name",
    [
        "user_idx",
        "item_idx",
        "target",
    ],
)
def test_dataset_rejects_null_required_values(
    interactions: pd.DataFrame,
    column_name: str,
) -> None:
    invalid_interactions = interactions.copy()
    invalid_interactions.loc[0, column_name] = None

    with pytest.raises(
        ValueError,
        match="must not contain null values",
    ):
        ImplicitFeedbackDataset(invalid_interactions)


@pytest.mark.parametrize(
    "column_name",
    [
        "user_idx",
        "item_idx",
    ],
)
def test_dataset_rejects_non_integer_indices(
    interactions: pd.DataFrame,
    column_name: str,
) -> None:
    invalid_interactions = interactions.copy()
    invalid_interactions[column_name] = invalid_interactions[column_name].astype(float)

    with pytest.raises(
        ValueError,
        match=f"{column_name} must contain integer values",
    ):
        ImplicitFeedbackDataset(invalid_interactions)


@pytest.mark.parametrize(
    "column_name",
    [
        "user_idx",
        "item_idx",
    ],
)
def test_dataset_rejects_negative_indices(
    interactions: pd.DataFrame,
    column_name: str,
) -> None:
    invalid_interactions = interactions.copy()
    invalid_interactions.loc[0, column_name] = -1

    with pytest.raises(
        ValueError,
        match=f"{column_name} must contain non-negative values",
    ):
        ImplicitFeedbackDataset(invalid_interactions)


def test_dataset_rejects_non_binary_targets(
    interactions: pd.DataFrame,
) -> None:
    invalid_interactions = interactions.copy()
    invalid_interactions.loc[0, "target"] = 2

    with pytest.raises(
        ValueError,
        match="target must contain only binary values",
    ):
        ImplicitFeedbackDataset(invalid_interactions)
