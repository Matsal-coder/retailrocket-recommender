"""Unit tests for the popularity recommender."""

from __future__ import annotations

import pickle
from pathlib import Path

import pandas as pd
import pytest

from retail_recommender.models.popularity import (
    PopularityRecommender,
)


@pytest.fixture
def interactions() -> pd.DataFrame:
    """Create deterministic interaction data for popularity tests."""
    return pd.DataFrame(
        {
            "user_idx": [0, 0, 1, 1, 2],
            "item_idx": [10, 20, 10, 30, 20],
            "interaction_score": [5.0, 1.0, 3.0, 4.0, 2.0],
            "interaction_count": [1, 1, 2, 1, 3],
        }
    )


def test_fit_orders_items_by_aggregated_popularity(
    interactions: pd.DataFrame,
) -> None:
    model = PopularityRecommender()

    fitted_model = model.fit(interactions)

    assert fitted_model is model
    assert model.is_fitted
    assert model.ranked_items == [10, 30, 20]


def test_fit_uses_interaction_count_as_secondary_criterion() -> None:
    interactions = pd.DataFrame(
        {
            "item_idx": [10, 20],
            "interaction_score": [5.0, 5.0],
            "interaction_count": [1, 3],
        }
    )
    model = PopularityRecommender()

    model.fit(interactions)

    assert model.ranked_items == [20, 10]


def test_fit_uses_item_id_as_deterministic_tiebreaker() -> None:
    interactions = pd.DataFrame(
        {
            "item_idx": [20, 10],
            "interaction_score": [5.0, 5.0],
            "interaction_count": [1, 1],
        }
    )
    model = PopularityRecommender()

    model.fit(interactions)

    assert model.ranked_items == [10, 20]


def test_recommend_returns_top_k_items(
    interactions: pd.DataFrame,
) -> None:
    model = PopularityRecommender().fit(interactions)

    recommendations = model.recommend(user_id=999, k=2)

    assert recommendations == [10, 30]


def test_recommend_excludes_seen_items(
    interactions: pd.DataFrame,
) -> None:
    model = PopularityRecommender().fit(interactions)

    recommendations = model.recommend(
        user_id=0,
        k=2,
        seen_items={10},
    )

    assert recommendations == [30, 20]


def test_recommend_returns_available_items_when_k_is_larger(
    interactions: pd.DataFrame,
) -> None:
    model = PopularityRecommender().fit(interactions)

    recommendations = model.recommend(user_id=0, k=10)

    assert recommendations == [10, 30, 20]


def test_recommend_rejects_non_positive_k(
    interactions: pd.DataFrame,
) -> None:
    model = PopularityRecommender().fit(interactions)

    with pytest.raises(
        ValueError,
        match="k must be greater than zero",
    ):
        model.recommend(user_id=0, k=0)


def test_recommend_rejects_unfitted_model() -> None:
    model = PopularityRecommender()

    with pytest.raises(
        RuntimeError,
        match="must be fitted before use",
    ):
        model.recommend(user_id=0, k=2)


def test_fit_rejects_non_dataframe() -> None:
    model = PopularityRecommender()

    with pytest.raises(
        TypeError,
        match="must be a pandas DataFrame",
    ):
        model.fit([(0, 10)])


def test_fit_rejects_empty_dataframe() -> None:
    interactions = pd.DataFrame(
        columns=[
            "item_idx",
            "interaction_score",
            "interaction_count",
        ]
    )
    model = PopularityRecommender()

    with pytest.raises(
        ValueError,
        match="must not be empty",
    ):
        model.fit(interactions)


def test_fit_rejects_missing_columns() -> None:
    interactions = pd.DataFrame(
        {
            "item_idx": [10],
            "interaction_score": [1.0],
        }
    )
    model = PopularityRecommender()

    with pytest.raises(
        ValueError,
        match="interaction_count",
    ):
        model.fit(interactions)


def test_save_and_load_preserve_ranking(
    interactions: pd.DataFrame,
    tmp_path: Path,
) -> None:
    model_path = tmp_path / "popularity.pkl"
    model = PopularityRecommender().fit(interactions)

    model.save(model_path)
    loaded_model = PopularityRecommender.load(model_path)

    assert loaded_model.is_fitted
    assert loaded_model.ranked_items == model.ranked_items
    assert loaded_model.recommend(user_id=0, k=2) == [10, 30]


def test_save_creates_parent_directory(
    interactions: pd.DataFrame,
    tmp_path: Path,
) -> None:
    model_path = tmp_path / "nested" / "popularity.pkl"
    model = PopularityRecommender().fit(interactions)

    model.save(model_path)

    assert model_path.exists()


def test_save_rejects_unfitted_model(tmp_path: Path) -> None:
    model_path = tmp_path / "popularity.pkl"
    model = PopularityRecommender()

    with pytest.raises(
        RuntimeError,
        match="must be fitted before use",
    ):
        model.save(model_path)


def test_load_rejects_missing_file(tmp_path: Path) -> None:
    model_path = tmp_path / "missing.pkl"

    with pytest.raises(
        FileNotFoundError,
        match="Model file not found",
    ):
        PopularityRecommender.load(model_path)


def test_load_rejects_invalid_payload(tmp_path: Path) -> None:
    model_path = tmp_path / "invalid.pkl"

    with model_path.open("wb") as model_file:
        pickle.dump(["invalid"], model_file)

    with pytest.raises(
        ValueError,
        match="Invalid popularity model payload",
    ):
        PopularityRecommender.load(model_path)
