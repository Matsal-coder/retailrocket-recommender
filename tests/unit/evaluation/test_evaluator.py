"""Unit tests for the recommender evaluator."""

from __future__ import annotations

from collections.abc import Collection

import pytest

from retail_recommender.evaluation.evaluator import (
    RecommenderEvaluator,
)

DEFAULT_K = 2
CATALOG_ITEMS = {10, 20, 30, 40}
EVALUATED_USERS = 2


class StaticRecommender:
    """Return predefined recommendations for tests."""

    def __init__(
        self,
        recommendations_by_user: dict[int, list[int]],
    ) -> None:
        self.recommendations_by_user = recommendations_by_user
        self.received_seen_items: dict[int, set[int]] = {}

    def recommend(
        self,
        user_id: int,
        k: int,
        seen_items: Collection[int] | None = None,
    ) -> list[int]:
        """Return static recommendations excluding seen items."""
        excluded_items = set(seen_items or [])
        self.received_seen_items[user_id] = excluded_items

        recommendations = self.recommendations_by_user.get(
            user_id,
            [],
        )

        return [item for item in recommendations if item not in excluded_items][:k]


def test_evaluator_aggregates_user_metrics() -> None:
    recommender = StaticRecommender(
        {
            1: [10, 20],
            2: [30, 40],
        }
    )
    evaluator = RecommenderEvaluator(
        recommender,
        k=DEFAULT_K,
        catalog_items=CATALOG_ITEMS,
    )

    result = evaluator.evaluate(
        relevant_items_by_user={
            1: {10},
            2: {30},
        }
    )

    assert result.evaluated_users == EVALUATED_USERS
    assert result.precision_at_k == pytest.approx(0.5)
    assert result.recall_at_k == pytest.approx(1.0)
    assert result.ndcg_at_k == pytest.approx(1.0)
    assert result.map_at_k == pytest.approx(1.0)
    assert result.coverage_at_k == pytest.approx(1.0)


def test_evaluator_passes_seen_items_to_recommender() -> None:
    recommender = StaticRecommender(
        {
            1: [10, 20, 30],
        }
    )
    evaluator = RecommenderEvaluator(
        recommender,
        k=DEFAULT_K,
        catalog_items=CATALOG_ITEMS,
        exclude_seen_items=True,
    )

    result = evaluator.evaluate(
        relevant_items_by_user={
            1: {20},
        },
        seen_items_by_user={
            1: {10},
        },
    )

    assert recommender.received_seen_items[1] == {10}
    assert result.recall_at_k == pytest.approx(1.0)


def test_evaluator_can_keep_seen_items() -> None:
    recommender = StaticRecommender(
        {
            1: [10, 20],
        }
    )
    evaluator = RecommenderEvaluator(
        recommender,
        k=DEFAULT_K,
        catalog_items=CATALOG_ITEMS,
        exclude_seen_items=False,
    )

    evaluator.evaluate(
        relevant_items_by_user={
            1: {20},
        },
        seen_items_by_user={
            1: {10},
        },
    )

    assert recommender.received_seen_items[1] == set()


def test_evaluator_ignores_users_without_relevant_items() -> None:
    recommender = StaticRecommender(
        {
            1: [10, 20],
            2: [30, 40],
        }
    )
    evaluator = RecommenderEvaluator(
        recommender,
        k=DEFAULT_K,
        catalog_items=CATALOG_ITEMS,
    )

    result = evaluator.evaluate(
        relevant_items_by_user={
            1: {10},
            2: set(),
        }
    )

    assert result.evaluated_users == 1


def test_evaluator_result_can_be_serialized() -> None:
    recommender = StaticRecommender(
        {
            1: [10, 20],
        }
    )
    evaluator = RecommenderEvaluator(
        recommender,
        k=DEFAULT_K,
        catalog_items=CATALOG_ITEMS,
    )

    result = evaluator.evaluate(
        relevant_items_by_user={
            1: {10},
        }
    )

    serialized_result = result.to_dict()

    assert serialized_result["k"] == DEFAULT_K
    assert serialized_result["evaluated_users"] == 1
    assert "ndcg_at_k" in serialized_result


def test_evaluator_rejects_non_positive_k() -> None:
    recommender = StaticRecommender({})

    with pytest.raises(
        ValueError,
        match="k must be greater than zero",
    ):
        RecommenderEvaluator(
            recommender,
            k=0,
            catalog_items=CATALOG_ITEMS,
        )


def test_evaluator_rejects_empty_catalog() -> None:
    recommender = StaticRecommender({})

    with pytest.raises(
        ValueError,
        match="catalog_items must not be empty",
    ):
        RecommenderEvaluator(
            recommender,
            k=DEFAULT_K,
            catalog_items=set(),
        )


def test_evaluator_rejects_empty_ground_truth() -> None:
    recommender = StaticRecommender({})
    evaluator = RecommenderEvaluator(
        recommender,
        k=DEFAULT_K,
        catalog_items=CATALOG_ITEMS,
    )

    with pytest.raises(
        ValueError,
        match="must contain evaluable users",
    ):
        evaluator.evaluate(
            relevant_items_by_user={
                1: set(),
            }
        )
