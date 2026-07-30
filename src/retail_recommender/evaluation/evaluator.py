"""Top-K evaluator for recommendation models."""

from __future__ import annotations

from collections.abc import Collection, Mapping
from dataclasses import asdict, dataclass
from typing import Protocol

from retail_recommender.evaluation.ranking_metrics import (
    coverage_at_k,
    map_at_k,
    ndcg_at_k,
    precision_at_k,
    recall_at_k,
)


class RecommendationProvider(Protocol):
    """Contract required by the Top-K evaluator."""

    def recommend(
        self,
        user_id: int,
        k: int,
        seen_items: Collection[int] | None = None,
    ) -> list[int]:
        """Return ranked recommendations for one user."""


@dataclass(frozen=True)
class EvaluationResult:
    """Aggregated Top-K evaluation result."""

    k: int
    evaluated_users: int
    precision_at_k: float
    recall_at_k: float
    ndcg_at_k: float
    map_at_k: float
    coverage_at_k: float

    def to_dict(self) -> dict[str, int | float]:
        """Serialize evaluation metrics to a dictionary."""
        return asdict(self)


class RecommenderEvaluator:
    """Evaluate a recommendation provider across multiple users."""

    def __init__(
        self,
        recommender: RecommendationProvider,
        *,
        k: int,
        catalog_items: Collection[int],
        exclude_seen_items: bool = True,
    ) -> None:
        """Initialize the evaluator.

        Args:
            recommender: Object capable of producing Top-K recommendations.
            k: Recommendation cutoff.
            catalog_items: Complete candidate item catalog.
            exclude_seen_items: Whether training items should be excluded.

        Raises:
            ValueError: If K or the catalog is invalid.
        """
        if k <= 0:
            msg = "k must be greater than zero"
            raise ValueError(msg)

        catalog_set = set(catalog_items)

        if not catalog_set:
            msg = "catalog_items must not be empty"
            raise ValueError(msg)

        self.recommender = recommender
        self.k = k
        self.catalog_items = catalog_set
        self.exclude_seen_items = exclude_seen_items

    def evaluate(
        self,
        relevant_items_by_user: Mapping[int, Collection[int]],
        seen_items_by_user: Mapping[int, Collection[int]] | None = None,
    ) -> EvaluationResult:
        """Evaluate recommendations against relevant test items.

        Args:
            relevant_items_by_user: Ground-truth relevant items per user.
            seen_items_by_user: Training items already seen per user.

        Returns:
            Aggregated Top-K metrics.

        Raises:
            ValueError: If no evaluable users are provided.
        """
        evaluation_users = [
            user_id
            for user_id, relevant_items in relevant_items_by_user.items()
            if relevant_items
        ]

        if not evaluation_users:
            msg = "relevant_items_by_user must contain evaluable users"
            raise ValueError(msg)

        seen_mapping = seen_items_by_user or {}

        recommendations_by_user: dict[int, list[int]] = {}
        precision_values: list[float] = []
        recall_values: list[float] = []
        ndcg_values: list[float] = []
        map_values: list[float] = []

        for user_id in evaluation_users:
            relevant_items = relevant_items_by_user[user_id]
            seen_items = (
                seen_mapping.get(user_id, set()) if self.exclude_seen_items else None
            )

            recommendations = self.recommender.recommend(
                user_id=user_id,
                k=self.k,
                seen_items=seen_items,
            )

            recommendations_by_user[user_id] = recommendations

            precision_values.append(
                precision_at_k(
                    recommendations,
                    relevant_items,
                    self.k,
                )
            )
            recall_values.append(
                recall_at_k(
                    recommendations,
                    relevant_items,
                    self.k,
                )
            )
            ndcg_values.append(
                ndcg_at_k(
                    recommendations,
                    relevant_items,
                    self.k,
                )
            )
            map_values.append(
                map_at_k(
                    recommendations,
                    relevant_items,
                    self.k,
                )
            )

        evaluated_user_count = len(evaluation_users)

        return EvaluationResult(
            k=self.k,
            evaluated_users=evaluated_user_count,
            precision_at_k=sum(precision_values) / evaluated_user_count,
            recall_at_k=sum(recall_values) / evaluated_user_count,
            ndcg_at_k=sum(ndcg_values) / evaluated_user_count,
            map_at_k=sum(map_values) / evaluated_user_count,
            coverage_at_k=coverage_at_k(
                recommendations_by_user,
                self.catalog_items,
                self.k,
            ),
        )
