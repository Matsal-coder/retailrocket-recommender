"""Unit tests for Neural Collaborative Filtering."""

from __future__ import annotations

from typing import Any

import pytest
import torch
from torch import nn

from retail_recommender.models.neural_cf import (
    NeuralCollaborativeFiltering,
)

NUM_USERS = 5
NUM_ITEMS = 8
EMBEDDING_DIMENSION = 4
FIRST_HIDDEN_DIMENSION = 8
SECOND_HIDDEN_DIMENSION = 4
BATCH_SIZE = 3
DROPOUT_PROBABILITY = 0.2


@pytest.fixture
def model() -> NeuralCollaborativeFiltering:
    """Create a small deterministic Neural CF architecture."""
    torch.manual_seed(123)

    return NeuralCollaborativeFiltering(
        num_users=NUM_USERS,
        num_items=NUM_ITEMS,
        embedding_dim=EMBEDDING_DIMENSION,
        hidden_layers=(
            FIRST_HIDDEN_DIMENSION,
            SECOND_HIDDEN_DIMENSION,
        ),
        dropout=DROPOUT_PROBABILITY,
    )


def test_model_creates_expected_embedding_shapes(
    model: NeuralCollaborativeFiltering,
) -> None:
    assert model.user_embedding.weight.shape == (
        NUM_USERS,
        EMBEDDING_DIMENSION,
    )
    assert model.item_embedding.weight.shape == (
        NUM_ITEMS,
        EMBEDDING_DIMENSION,
    )


def test_model_stores_configuration(
    model: NeuralCollaborativeFiltering,
) -> None:
    assert model.num_users == NUM_USERS
    assert model.num_items == NUM_ITEMS
    assert model.embedding_dim == EMBEDDING_DIMENSION
    assert model.hidden_layers == (
        FIRST_HIDDEN_DIMENSION,
        SECOND_HIDDEN_DIMENSION,
    )
    assert model.dropout == DROPOUT_PROBABILITY


def test_mlp_has_expected_linear_dimensions(
    model: NeuralCollaborativeFiltering,
) -> None:
    linear_layers = [layer for layer in model.mlp if isinstance(layer, nn.Linear)]

    assert len(linear_layers) == BATCH_SIZE
    assert linear_layers[0].in_features == EMBEDDING_DIMENSION * 2
    assert linear_layers[0].out_features == FIRST_HIDDEN_DIMENSION
    assert linear_layers[1].in_features == FIRST_HIDDEN_DIMENSION
    assert linear_layers[1].out_features == SECOND_HIDDEN_DIMENSION
    assert linear_layers[2].in_features == SECOND_HIDDEN_DIMENSION
    assert linear_layers[2].out_features == 1


def test_forward_returns_one_logit_per_pair(
    model: NeuralCollaborativeFiltering,
) -> None:
    user_indices = torch.tensor(
        [0, 1, 2],
        dtype=torch.long,
    )
    item_indices = torch.tensor(
        [3, 4, 5],
        dtype=torch.long,
    )

    logits = model(user_indices, item_indices)

    assert logits.shape == (BATCH_SIZE,)
    assert logits.dtype == torch.float32


def test_forward_supports_single_pair(
    model: NeuralCollaborativeFiltering,
) -> None:
    user_indices = torch.tensor([0], dtype=torch.long)
    item_indices = torch.tensor([1], dtype=torch.long)

    logits = model(user_indices, item_indices)

    assert logits.shape == (1,)


def test_forward_produces_gradients(
    model: NeuralCollaborativeFiltering,
) -> None:
    user_indices = torch.tensor(
        [0, 1, 2],
        dtype=torch.long,
    )
    item_indices = torch.tensor(
        [3, 4, 5],
        dtype=torch.long,
    )

    logits = model(user_indices, item_indices)
    loss = logits.mean()
    loss.backward()

    assert model.user_embedding.weight.grad is not None
    assert model.item_embedding.weight.grad is not None


def test_model_eval_mode_disables_dropout(
    model: NeuralCollaborativeFiltering,
) -> None:
    model.eval()

    dropout_layers = [layer for layer in model.mlp if isinstance(layer, nn.Dropout)]

    assert dropout_layers
    assert all(not layer.training for layer in dropout_layers)


@pytest.mark.parametrize(
    ("parameter_name", "parameter_value", "expected_message"),
    [
        ("num_users", 0, "num_users must be greater than zero"),
        ("num_items", 0, "num_items must be greater than zero"),
        (
            "embedding_dim",
            0,
            "embedding_dim must be greater than zero",
        ),
    ],
)
def test_constructor_rejects_non_positive_dimensions(
    parameter_name: str,
    parameter_value: int,
    expected_message: str,
) -> None:
    parameters: dict[str, Any] = {
        "num_users": NUM_USERS,
        "num_items": NUM_ITEMS,
        "embedding_dim": EMBEDDING_DIMENSION,
        "hidden_layers": (FIRST_HIDDEN_DIMENSION,),
        "dropout": DROPOUT_PROBABILITY,
    }
    parameters[parameter_name] = parameter_value

    with pytest.raises(ValueError, match=expected_message):
        NeuralCollaborativeFiltering(**parameters)


def test_constructor_rejects_empty_hidden_layers() -> None:
    with pytest.raises(
        ValueError,
        match="hidden_layers must contain at least one layer",
    ):
        NeuralCollaborativeFiltering(
            num_users=NUM_USERS,
            num_items=NUM_ITEMS,
            embedding_dim=EMBEDDING_DIMENSION,
            hidden_layers=(),
        )


def test_constructor_rejects_non_positive_hidden_layer() -> None:
    with pytest.raises(
        ValueError,
        match="hidden layer dimensions must be greater than zero",
    ):
        NeuralCollaborativeFiltering(
            num_users=NUM_USERS,
            num_items=NUM_ITEMS,
            embedding_dim=EMBEDDING_DIMENSION,
            hidden_layers=(FIRST_HIDDEN_DIMENSION, 0),
        )


@pytest.mark.parametrize(
    "dropout",
    [-0.1, 1.0, 1.1],
)
def test_constructor_rejects_invalid_dropout(
    dropout: float,
) -> None:
    with pytest.raises(
        ValueError,
        match=("dropout must be greater than or equal to zero " "and less than one"),
    ):
        NeuralCollaborativeFiltering(
            num_users=NUM_USERS,
            num_items=NUM_ITEMS,
            dropout=dropout,
        )


def test_forward_rejects_different_input_shapes(
    model: NeuralCollaborativeFiltering,
) -> None:
    user_indices = torch.tensor([0, 1], dtype=torch.long)
    item_indices = torch.tensor([1], dtype=torch.long)

    with pytest.raises(
        ValueError,
        match="must have the same shape",
    ):
        model(user_indices, item_indices)


def test_forward_rejects_multidimensional_user_tensor(
    model: NeuralCollaborativeFiltering,
) -> None:
    user_indices = torch.tensor(
        [[0, 1]],
        dtype=torch.long,
    )
    item_indices = torch.tensor(
        [1, 2],
        dtype=torch.long,
    )

    with pytest.raises(
        ValueError,
        match="user_indices must be a one-dimensional tensor",
    ):
        model(user_indices, item_indices)


def test_forward_rejects_multidimensional_item_tensor(
    model: NeuralCollaborativeFiltering,
) -> None:
    user_indices = torch.tensor(
        [0, 1],
        dtype=torch.long,
    )
    item_indices = torch.tensor(
        [[1, 2]],
        dtype=torch.long,
    )

    with pytest.raises(
        ValueError,
        match="item_indices must be a one-dimensional tensor",
    ):
        model(user_indices, item_indices)


def test_forward_rejects_non_long_user_indices(
    model: NeuralCollaborativeFiltering,
) -> None:
    user_indices = torch.tensor(
        [0.0, 1.0],
        dtype=torch.float32,
    )
    item_indices = torch.tensor(
        [1, 2],
        dtype=torch.long,
    )

    with pytest.raises(
        TypeError,
        match="user_indices must use torch.long dtype",
    ):
        model(user_indices, item_indices)


def test_forward_rejects_non_long_item_indices(
    model: NeuralCollaborativeFiltering,
) -> None:
    user_indices = torch.tensor(
        [0, 1],
        dtype=torch.long,
    )
    item_indices = torch.tensor(
        [1.0, 2.0],
        dtype=torch.float32,
    )

    with pytest.raises(
        TypeError,
        match="item_indices must use torch.long dtype",
    ):
        model(user_indices, item_indices)


def test_forward_rejects_empty_tensors(
    model: NeuralCollaborativeFiltering,
) -> None:
    user_indices = torch.tensor([], dtype=torch.long)
    item_indices = torch.tensor([], dtype=torch.long)

    with pytest.raises(
        ValueError,
        match="input tensors must not be empty",
    ):
        model(user_indices, item_indices)


@pytest.mark.parametrize(
    "invalid_user_index",
    [-1, NUM_USERS],
)
def test_forward_rejects_user_index_outside_range(
    model: NeuralCollaborativeFiltering,
    invalid_user_index: int,
) -> None:
    user_indices = torch.tensor(
        [invalid_user_index],
        dtype=torch.long,
    )
    item_indices = torch.tensor([0], dtype=torch.long)

    with pytest.raises(
        IndexError,
        match="user index is outside",
    ):
        model(user_indices, item_indices)


@pytest.mark.parametrize(
    "invalid_item_index",
    [-1, NUM_ITEMS],
)
def test_forward_rejects_item_index_outside_range(
    model: NeuralCollaborativeFiltering,
    invalid_item_index: int,
) -> None:
    user_indices = torch.tensor([0], dtype=torch.long)
    item_indices = torch.tensor(
        [invalid_item_index],
        dtype=torch.long,
    )

    with pytest.raises(
        IndexError,
        match="item index is outside",
    ):
        model(user_indices, item_indices)
