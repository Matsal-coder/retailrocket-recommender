"""Neural Collaborative Filtering model implemented with PyTorch."""

from __future__ import annotations

from collections.abc import Sequence

import torch
from torch import Tensor, nn

MINIMUM_ENTITY_COUNT = 1
MINIMUM_EMBEDDING_DIMENSION = 1
MINIMUM_HIDDEN_DIMENSION = 1


class NeuralCollaborativeFiltering(nn.Module):
    """Estimate affinity logits for encoded user-item pairs."""

    def __init__(
        self,
        num_users: int,
        num_items: int,
        embedding_dim: int = 32,
        hidden_layers: Sequence[int] = (64, 32),
        dropout: float = 0.2,
    ) -> None:
        """Initialize the neural collaborative filtering architecture.

        Args:
            num_users: Number of encoded users.
            num_items: Number of encoded items.
            embedding_dim: Size of user and item latent representations.
            hidden_layers: Number of neurons in each hidden MLP layer.
            dropout: Dropout probability applied after hidden activations.

        Raises:
            ValueError: If an architecture parameter is invalid.
        """
        super().__init__()

        self._validate_configuration(
            num_users=num_users,
            num_items=num_items,
            embedding_dim=embedding_dim,
            hidden_layers=hidden_layers,
            dropout=dropout,
        )

        self.num_users = num_users
        self.num_items = num_items
        self.embedding_dim = embedding_dim
        self.hidden_layers = tuple(hidden_layers)
        self.dropout = dropout

        self.user_embedding = nn.Embedding(
            num_embeddings=num_users,
            embedding_dim=embedding_dim,
        )
        self.item_embedding = nn.Embedding(
            num_embeddings=num_items,
            embedding_dim=embedding_dim,
        )

        mlp_input_dimension = embedding_dim * 2
        self.mlp = self._build_mlp(
            input_dimension=mlp_input_dimension,
            hidden_layers=self.hidden_layers,
            dropout=dropout,
        )

        self._initialize_parameters()

    def forward(
        self,
        user_indices: Tensor,
        item_indices: Tensor,
    ) -> Tensor:
        """Calculate affinity logits for user-item pairs.

        Args:
            user_indices: One-dimensional tensor of encoded user indices.
            item_indices: One-dimensional tensor of encoded item indices.

        Returns:
            One-dimensional tensor containing one logit per pair.

        Raises:
            ValueError: If input tensors have incompatible shapes.
            TypeError: If input tensors do not contain integer indices.
            IndexError: If user or item indices are outside embedding ranges.
        """
        self._validate_forward_inputs(
            user_indices=user_indices,
            item_indices=item_indices,
        )

        user_vectors = self.user_embedding(user_indices)
        item_vectors = self.item_embedding(item_indices)

        combined_vectors = torch.cat(
            [user_vectors, item_vectors],
            dim=1,
        )

        logits = self.mlp(combined_vectors)

        return logits.squeeze(-1)

    @staticmethod
    def _build_mlp(
        input_dimension: int,
        hidden_layers: Sequence[int],
        dropout: float,
    ) -> nn.Sequential:
        """Build the multilayer perceptron used for affinity scoring."""
        layers: list[nn.Module] = []
        current_dimension = input_dimension

        for hidden_dimension in hidden_layers:
            layers.extend(
                [
                    nn.Linear(
                        current_dimension,
                        hidden_dimension,
                    ),
                    nn.ReLU(),
                    nn.Dropout(dropout),
                ]
            )
            current_dimension = hidden_dimension

        layers.append(nn.Linear(current_dimension, 1))

        return nn.Sequential(*layers)

    def _initialize_parameters(self) -> None:
        """Initialize embedding and linear layer parameters."""
        nn.init.normal_(
            self.user_embedding.weight,
            mean=0.0,
            std=0.01,
        )
        nn.init.normal_(
            self.item_embedding.weight,
            mean=0.0,
            std=0.01,
        )

        for module in self.mlp:
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.zeros_(module.bias)

    @staticmethod
    def _validate_configuration(
        num_users: int,
        num_items: int,
        embedding_dim: int,
        hidden_layers: Sequence[int],
        dropout: float,
    ) -> None:
        """Validate model architecture parameters."""
        if num_users < MINIMUM_ENTITY_COUNT:
            msg = "num_users must be greater than zero"
            raise ValueError(msg)

        if num_items < MINIMUM_ENTITY_COUNT:
            msg = "num_items must be greater than zero"
            raise ValueError(msg)

        if embedding_dim < MINIMUM_EMBEDDING_DIMENSION:
            msg = "embedding_dim must be greater than zero"
            raise ValueError(msg)

        if not hidden_layers:
            msg = "hidden_layers must contain at least one layer"
            raise ValueError(msg)

        if any(
            hidden_dimension < MINIMUM_HIDDEN_DIMENSION
            for hidden_dimension in hidden_layers
        ):
            msg = "hidden layer dimensions must be greater than zero"
            raise ValueError(msg)

        if not 0.0 <= dropout < 1.0:
            msg = "dropout must be greater than or equal to zero and less than one"
            raise ValueError(msg)

    def _validate_forward_inputs(
        self,
        user_indices: Tensor,
        item_indices: Tensor,
    ) -> None:
        """Validate input tensors before embedding lookup."""
        if user_indices.ndim != 1:
            msg = "user_indices must be a one-dimensional tensor"
            raise ValueError(msg)

        if item_indices.ndim != 1:
            msg = "item_indices must be a one-dimensional tensor"
            raise ValueError(msg)

        if user_indices.shape != item_indices.shape:
            msg = "user_indices and item_indices must have the same shape"
            raise ValueError(msg)

        if user_indices.dtype != torch.long:
            msg = "user_indices must use torch.long dtype"
            raise TypeError(msg)

        if item_indices.dtype != torch.long:
            msg = "item_indices must use torch.long dtype"
            raise TypeError(msg)

        if user_indices.numel() == 0:
            msg = "input tensors must not be empty"
            raise ValueError(msg)

        if torch.any(user_indices < 0) or torch.any(user_indices >= self.num_users):
            msg = "user index is outside the configured embedding range"
            raise IndexError(msg)

        if torch.any(item_indices < 0) or torch.any(item_indices >= self.num_items):
            msg = "item index is outside the configured embedding range"
            raise IndexError(msg)
