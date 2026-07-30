"""Unit tests for the Neural CF recommendation adapter."""

import torch
from torch import nn

from retail_recommender.evaluation.neural_recommender import (
    NeuralCFRecommender,
)

DEFAULT_K = 2
NUM_ITEMS = 4


class DeterministicModel(nn.Module):
    """Return the item ID as its ranking score."""

    def forward(
        self,
        user_ids: torch.Tensor,
        item_ids: torch.Tensor,
    ) -> torch.Tensor:
        del user_ids
        return item_ids.to(dtype=torch.float32)


def test_neural_recommender_returns_highest_scoring_items() -> None:
    recommender = NeuralCFRecommender(
        model=DeterministicModel(),
        num_items=NUM_ITEMS,
        device=torch.device("cpu"),
        candidate_batch_size=2,
    )

    recommendations = recommender.recommend(
        user_id=0,
        k=DEFAULT_K,
    )

    assert recommendations == [3, 2]


def test_neural_recommender_excludes_seen_items() -> None:
    recommender = NeuralCFRecommender(
        model=DeterministicModel(),
        num_items=NUM_ITEMS,
        device=torch.device("cpu"),
        candidate_batch_size=2,
    )

    recommendations = recommender.recommend(
        user_id=0,
        k=DEFAULT_K,
        seen_items={3},
    )

    assert recommendations == [2, 1]


def test_neural_recommender_returns_empty_for_full_history() -> None:
    recommender = NeuralCFRecommender(
        model=DeterministicModel(),
        num_items=NUM_ITEMS,
        device=torch.device("cpu"),
        candidate_batch_size=2,
    )

    recommendations = recommender.recommend(
        user_id=0,
        k=DEFAULT_K,
        seen_items={0, 1, 2, 3},
    )

    assert recommendations == []
