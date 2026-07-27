"""Pure Top-K ranking metrics for recommender systems."""

from __future__ import annotations

import math
from collections.abc import Collection, Hashable, Mapping, Sequence
from typing import TypeVar

ItemId = TypeVar("ItemId", bound=Hashable)
UserId = TypeVar("UserId", bound=Hashable)


def precision_at_k(
    recommended_items: Sequence[ItemId],
    relevant_items: Collection[ItemId],
    k: int,
) -> float:
    """Calculate Precision@K for one user.

    Precision@K is the number of relevant items found among the first K
    recommendations divided by K.

    Args:
        recommended_items: Ranked item identifiers recommended to the user.
        relevant_items: Ground-truth relevant item identifiers.
        k: Maximum number of recommendations to evaluate.

    Returns:
        Precision@K value between 0.0 and 1.0.

    Raises:
        ValueError: If K is not greater than zero.
    """
    _validate_k(k)

    top_k_items = _unique_top_k(recommended_items, k)
    relevant_set = set(relevant_items)
    relevant_recommendations = sum(item in relevant_set for item in top_k_items)

    return relevant_recommendations / k


def recall_at_k(
    recommended_items: Sequence[ItemId],
    relevant_items: Collection[ItemId],
    k: int,
) -> float:
    """Calculate Recall@K for one user.

    Recall@K is the number of relevant items found among the first K
    recommendations divided by the total number of relevant items.

    Args:
        recommended_items: Ranked item identifiers recommended to the user.
        relevant_items: Ground-truth relevant item identifiers.
        k: Maximum number of recommendations to evaluate.

    Returns:
        Recall@K value between 0.0 and 1.0.

    Raises:
        ValueError: If K is not greater than zero.
    """
    _validate_k(k)

    relevant_set = set(relevant_items)
    if not relevant_set:
        return 0.0

    top_k_items = _unique_top_k(recommended_items, k)
    relevant_recommendations = sum(item in relevant_set for item in top_k_items)

    return relevant_recommendations / len(relevant_set)


def ndcg_at_k(
    recommended_items: Sequence[ItemId],
    relevant_items: Collection[ItemId],
    k: int,
) -> float:
    """Calculate binary Normalized Discounted Cumulative Gain at K.

    Relevant items receive a binary relevance score of one. Higher-ranked
    relevant items contribute more to the final score.

    Args:
        recommended_items: Ranked item identifiers recommended to the user.
        relevant_items: Ground-truth relevant item identifiers.
        k: Maximum number of recommendations to evaluate.

    Returns:
        NDCG@K value between 0.0 and 1.0.

    Raises:
        ValueError: If K is not greater than zero.
    """
    _validate_k(k)

    relevant_set = set(relevant_items)
    if not relevant_set:
        return 0.0

    top_k_items = _unique_top_k(recommended_items, k)

    discounted_cumulative_gain = sum(
        1.0 / math.log2(position + 2)
        for position, item in enumerate(top_k_items)
        if item in relevant_set
    )

    ideal_relevant_count = min(len(relevant_set), k)
    ideal_discounted_cumulative_gain = sum(
        1.0 / math.log2(position + 2) for position in range(ideal_relevant_count)
    )

    if ideal_discounted_cumulative_gain == 0.0:
        return 0.0

    return discounted_cumulative_gain / ideal_discounted_cumulative_gain


def map_at_k(
    recommended_items: Sequence[ItemId],
    relevant_items: Collection[ItemId],
    k: int,
) -> float:
    """Calculate Average Precision@K for one user.

    The project keeps the name ``map_at_k`` to match its public metric
    contract. The evaluator will average this result across users to produce
    the global Mean Average Precision.

    Args:
        recommended_items: Ranked item identifiers recommended to the user.
        relevant_items: Ground-truth relevant item identifiers.
        k: Maximum number of recommendations to evaluate.

    Returns:
        Average Precision@K value between 0.0 and 1.0.

    Raises:
        ValueError: If K is not greater than zero.
    """
    _validate_k(k)

    relevant_set = set(relevant_items)
    if not relevant_set:
        return 0.0

    top_k_items = _unique_top_k(recommended_items, k)
    relevant_items_found = 0
    precision_sum = 0.0

    for position, item in enumerate(top_k_items, start=1):
        if item not in relevant_set:
            continue

        relevant_items_found += 1
        precision_sum += relevant_items_found / position

    denominator = min(len(relevant_set), k)
    return precision_sum / denominator


def coverage_at_k(
    recommendations_by_user: Mapping[UserId, Sequence[ItemId]],
    catalog_items: Collection[ItemId],
    k: int,
) -> float:
    """Calculate catalog coverage across users at K.

    Coverage is the proportion of distinct catalog items appearing in at
    least one user's Top-K recommendation list.

    Args:
        recommendations_by_user: Ranked recommendations grouped by user.
        catalog_items: Complete collection of candidate catalog items.
        k: Maximum number of recommendations considered per user.

    Returns:
        Catalog coverage between 0.0 and 1.0.

    Raises:
        ValueError: If K is not greater than zero.
    """
    _validate_k(k)

    catalog_set = set(catalog_items)
    if not catalog_set:
        return 0.0

    recommended_catalog_items: set[ItemId] = set()

    for recommended_items in recommendations_by_user.values():
        top_k_items = _unique_top_k(recommended_items, k)
        recommended_catalog_items.update(
            item for item in top_k_items if item in catalog_set
        )

    return len(recommended_catalog_items) / len(catalog_set)


def _validate_k(k: int) -> None:
    """Validate the Top-K cutoff."""
    if k <= 0:
        msg = "k must be greater than zero"
        raise ValueError(msg)


def _unique_top_k(
    recommended_items: Sequence[ItemId],
    k: int,
) -> list[ItemId]:
    """Return the first K unique recommendations preserving ranking order."""
    unique_items: list[ItemId] = []
    seen_items: set[ItemId] = set()

    for item in recommended_items:
        if item in seen_items:
            continue

        seen_items.add(item)
        unique_items.append(item)

        if len(unique_items) == k:
            break

    return unique_items
