"""Unit tests for Top-K recommendation metrics."""

import math
from collections.abc import Callable, Collection, Sequence
from typing import TypeAlias

import pytest

from retail_recommender.evaluation.ranking_metrics import (
    coverage_at_k,
    map_at_k,
    ndcg_at_k,
    precision_at_k,
    recall_at_k,
)

MetricFunction: TypeAlias = Callable[
    [Sequence[int], Collection[int], int],
    float,
]

DEFAULT_K = 3


def test_precision_at_k_returns_relevant_fraction_over_k() -> None:
    recommended_items = [10, 20, 30]
    relevant_items = {10, 30, 40}

    result = precision_at_k(
        recommended_items,
        relevant_items,
        k=DEFAULT_K,
    )

    assert result == pytest.approx(2 / DEFAULT_K)


def test_precision_at_k_penalizes_incomplete_recommendation_list() -> None:
    recommended_items = [10]
    relevant_items = {10}

    result = precision_at_k(
        recommended_items,
        relevant_items,
        k=DEFAULT_K,
    )

    assert result == pytest.approx(1 / DEFAULT_K)


def test_recall_at_k_returns_recovered_relevant_fraction() -> None:
    recommended_items = [10, 20, 30]
    relevant_items = {10, 30, 40, 50}

    result = recall_at_k(
        recommended_items,
        relevant_items,
        k=DEFAULT_K,
    )

    assert result == pytest.approx(0.5)


def test_recall_at_k_returns_zero_when_no_relevant_items_exist() -> None:
    result = recall_at_k(
        recommended_items=[10, 20],
        relevant_items=set(),
        k=DEFAULT_K,
    )

    assert result == 0.0


def test_ndcg_at_k_returns_one_for_ideal_ranking() -> None:
    recommended_items = [10, 20, 30]
    relevant_items = {10, 20, 30}

    result = ndcg_at_k(
        recommended_items,
        relevant_items,
        k=DEFAULT_K,
    )

    assert result == pytest.approx(1.0)


def test_ndcg_at_k_penalizes_relevant_items_in_lower_positions() -> None:
    recommended_items = [99, 10, 20]
    relevant_items = {10, 20}

    result = ndcg_at_k(
        recommended_items,
        relevant_items,
        k=DEFAULT_K,
    )

    actual_dcg = (1 / math.log2(3)) + (1 / math.log2(4))
    ideal_dcg = (1 / math.log2(2)) + (1 / math.log2(3))
    expected = actual_dcg / ideal_dcg

    assert result == pytest.approx(expected)
    assert result < 1.0


def test_ndcg_at_k_returns_zero_without_relevant_items() -> None:
    result = ndcg_at_k(
        recommended_items=[10, 20],
        relevant_items=set(),
        k=DEFAULT_K,
    )

    assert result == 0.0


def test_map_at_k_averages_precision_at_relevant_positions() -> None:
    recommended_items = [10, 99, 20]
    relevant_items = {10, 20}

    result = map_at_k(
        recommended_items,
        relevant_items,
        k=DEFAULT_K,
    )

    expected = (1.0 + (2 / 3)) / 2
    assert result == pytest.approx(expected)


def test_map_at_k_returns_zero_without_relevant_items() -> None:
    result = map_at_k(
        recommended_items=[10, 20],
        relevant_items=set(),
        k=DEFAULT_K,
    )

    assert result == 0.0


def test_metrics_ignore_duplicate_recommendations() -> None:
    recommended_items = [10, 10, 20, 30]
    relevant_items = {10, 20}

    precision = precision_at_k(
        recommended_items,
        relevant_items,
        k=DEFAULT_K,
    )
    recall = recall_at_k(
        recommended_items,
        relevant_items,
        k=DEFAULT_K,
    )

    assert precision == pytest.approx(2 / DEFAULT_K)
    assert recall == pytest.approx(1.0)


def test_coverage_at_k_returns_catalog_fraction_recommended() -> None:
    recommendations_by_user = {
        1: [10, 20, 30],
        2: [20, 30, 40],
    }
    catalog_items = {10, 20, 30, 40, 50}

    result = coverage_at_k(
        recommendations_by_user,
        catalog_items,
        k=2,
    )

    assert result == pytest.approx(3 / 5)


def test_coverage_at_k_ignores_items_outside_catalog() -> None:
    recommendations_by_user = {
        1: [10, 999],
        2: [20, 888],
    }
    catalog_items = {10, 20, 30, 40}

    result = coverage_at_k(
        recommendations_by_user,
        catalog_items,
        k=2,
    )

    assert result == pytest.approx(0.5)


def test_coverage_at_k_returns_zero_for_empty_catalog() -> None:
    result = coverage_at_k(
        recommendations_by_user={1: [10, 20]},
        catalog_items=set(),
        k=DEFAULT_K,
    )

    assert result == 0.0


@pytest.mark.parametrize(
    "metric_function",
    [
        precision_at_k,
        recall_at_k,
        ndcg_at_k,
        map_at_k,
    ],
)
def test_user_metric_rejects_non_positive_k(
    metric_function: MetricFunction,
) -> None:
    with pytest.raises(ValueError, match="k must be greater than zero"):
        metric_function([10], {10}, 0)


def test_coverage_rejects_non_positive_k() -> None:
    with pytest.raises(ValueError, match="k must be greater than zero"):
        coverage_at_k(
            recommendations_by_user={1: [10]},
            catalog_items={10},
            k=0,
        )
