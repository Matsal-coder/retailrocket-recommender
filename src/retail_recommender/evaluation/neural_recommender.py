"""Recommendation adapter for the Neural CF model."""

from __future__ import annotations

from collections.abc import Collection

import torch

from retail_recommender.models.neural_cf import (
    NeuralCollaborativeFiltering,
)


class NeuralCFRecommender:
    """Generate Top-K recommendations from a trained Neural CF model."""

    def __init__(
        self,
        *,
        model: NeuralCollaborativeFiltering,
        num_items: int,
        device: torch.device,
        candidate_batch_size: int,
    ) -> None:
        """Initialize the neural recommendation adapter."""
        if num_items <= 0:
            msg = "num_items must be greater than zero"
            raise ValueError(msg)

        if candidate_batch_size <= 0:
            msg = "candidate_batch_size must be greater than zero"
            raise ValueError(msg)

        self.model = model
        self.num_items = num_items
        self.device = device
        self.candidate_batch_size = candidate_batch_size

        self.model.to(self.device)
        self.model.eval()

    def recommend(
        self,
        user_id: int,
        k: int,
        seen_items: Collection[int] | None = None,
    ) -> list[int]:
        """Recommend the highest-scoring unseen items for one user."""
        if user_id < 0:
            msg = "user_id must be non-negative"
            raise ValueError(msg)

        if k <= 0:
            msg = "k must be greater than zero"
            raise ValueError(msg)

        excluded_items = set(seen_items or ())
        candidate_items = [
            item_id
            for item_id in range(self.num_items)
            if item_id not in excluded_items
        ]

        if not candidate_items:
            return []

        scored_items: list[tuple[int, float]] = []

        with torch.no_grad():
            for batch_start in range(
                0,
                len(candidate_items),
                self.candidate_batch_size,
            ):
                batch_items = candidate_items[
                    batch_start : (batch_start + self.candidate_batch_size)
                ]

                item_tensor = torch.tensor(
                    batch_items,
                    dtype=torch.long,
                    device=self.device,
                )
                user_tensor = torch.full(
                    size=(len(batch_items),),
                    fill_value=user_id,
                    dtype=torch.long,
                    device=self.device,
                )

                logits = self.model(
                    user_tensor,
                    item_tensor,
                )
                scores = torch.sigmoid(logits)

                scored_items.extend(
                    zip(
                        batch_items,
                        scores.detach().cpu().tolist(),
                        strict=True,
                    )
                )

        scored_items.sort(
            key=lambda item_score: (
                -item_score[1],
                item_score[0],
            )
        )

        return [item_id for item_id, _ in scored_items[:k]]
