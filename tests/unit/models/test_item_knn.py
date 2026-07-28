"""Unit tests for the Item-KNN recommender."""

from __future__ import annotations

import pickle
from pathlib import Path

import pandas as pd
import pytest

from retail_recommender.models.item_knn import (
    ItemKNNRecommender,
)

ITEM_A = 10
ITEM_B = 20
ITEM_C = 30
ITEM_D = 40

DEFAULT_NEIGHBORS = 3


@pytest.fixture
def interactions() -> pd.DataFrame:
    """Create interactions with observable item similarity."""
    return pd.DataFrame(
        {
            "user_idx": [
                0,
                0,
                1,
                1,
                2,
                2,
                3,
            ],
            "item_idx": [
                10,
                20,
                10,
                20,
                20,
                30,
                40,
            ],
            "interaction_score": [
                5.0,
                3.0,
                4.0,
                3.0,
                2.0,
                5.0,
                1.0,
            ],
        }
    )


def test_fit_builds_item_neighbors(
    interactions: pd.DataFrame,
) -> None:
    model = ItemKNNRecommender(n_neighbors=2)

    fitted_model = model.fit(interactions)

    assert fitted_model is model
    assert model.is_fitted
    assert ITEM_A in model.item_neighbors
    assert ITEM_B in {item_id for item_id, _ in model.item_neighbors[10]}


def test_recommend_returns_similar_unseen_items(
    interactions: pd.DataFrame,
) -> None:
    model = ItemKNNRecommender(n_neighbors=3).fit(interactions)

    recommendations = model.recommend(user_id=2, k=2)

    assert ITEM_A in recommendations
    assert ITEM_B not in recommendations
    assert ITEM_C not in recommendations


def test_recommend_excludes_explicit_seen_items(
    interactions: pd.DataFrame,
) -> None:
    model = ItemKNNRecommender(n_neighbors=3).fit(interactions)

    recommendations = model.recommend(
        user_id=2,
        k=2,
        seen_items={10},
    )

    assert ITEM_A not in recommendations
    assert ITEM_B not in recommendations
    assert ITEM_C not in recommendations


def test_unknown_user_receives_popularity_fallback(
    interactions: pd.DataFrame,
) -> None:
    model = ItemKNNRecommender(n_neighbors=2).fit(interactions)

    recommendations = model.recommend(
        user_id=999,
        k=3,
    )

    assert recommendations == [10, 20, 30]


def test_recommend_returns_only_available_unseen_items(
    interactions: pd.DataFrame,
) -> None:
    model = ItemKNNRecommender(n_neighbors=2).fit(interactions)

    recommendations = model.recommend(
        user_id=0,
        k=10,
    )

    assert ITEM_A not in recommendations
    assert ITEM_B not in recommendations
    assert set(recommendations).issubset({30, 40})


def test_recommend_is_deterministic(
    interactions: pd.DataFrame,
) -> None:
    model = ItemKNNRecommender(n_neighbors=3).fit(interactions)

    first_result = model.recommend(user_id=2, k=2)
    second_result = model.recommend(user_id=2, k=2)

    assert first_result == second_result


def test_recommend_rejects_unfitted_model() -> None:
    model = ItemKNNRecommender()

    with pytest.raises(
        RuntimeError,
        match="must be fitted before use",
    ):
        model.recommend(user_id=0, k=2)


def test_recommend_rejects_non_positive_k(
    interactions: pd.DataFrame,
) -> None:
    model = ItemKNNRecommender().fit(interactions)

    with pytest.raises(
        ValueError,
        match="k must be greater than zero",
    ):
        model.recommend(user_id=0, k=0)


def test_constructor_rejects_invalid_neighbor_count() -> None:
    with pytest.raises(
        ValueError,
        match="n_neighbors must be greater than zero",
    ):
        ItemKNNRecommender(n_neighbors=0)


@pytest.mark.parametrize(
    "minimum_similarity",
    [-0.1, 1.1],
)
def test_constructor_rejects_invalid_minimum_similarity(
    minimum_similarity: float,
) -> None:
    with pytest.raises(
        ValueError,
        match="minimum_similarity must be between zero and one",
    ):
        ItemKNNRecommender(
            minimum_similarity=minimum_similarity,
        )


def test_fit_rejects_non_dataframe() -> None:
    model = ItemKNNRecommender()

    with pytest.raises(
        TypeError,
        match="must be a pandas DataFrame",
    ):
        model.fit([(0, 10, 1.0)])


def test_fit_rejects_empty_dataframe() -> None:
    interactions = pd.DataFrame(
        columns=[
            "user_idx",
            "item_idx",
            "interaction_score",
        ]
    )
    model = ItemKNNRecommender()

    with pytest.raises(
        ValueError,
        match="must not be empty",
    ):
        model.fit(interactions)


def test_fit_rejects_missing_columns() -> None:
    interactions = pd.DataFrame(
        {
            "user_idx": [0],
            "item_idx": [10],
        }
    )
    model = ItemKNNRecommender()

    with pytest.raises(
        ValueError,
        match="interaction_score",
    ):
        model.fit(interactions)


def test_save_and_load_preserve_recommendations(
    interactions: pd.DataFrame,
    tmp_path: Path,
) -> None:
    model_path = tmp_path / "item_knn.pkl"
    model = ItemKNNRecommender(
        n_neighbors=3,
        minimum_similarity=0.0,
    ).fit(interactions)

    expected = model.recommend(user_id=2, k=2)

    model.save(model_path)
    loaded_model = ItemKNNRecommender.load(model_path)

    assert loaded_model.is_fitted
    assert loaded_model.n_neighbors == DEFAULT_NEIGHBORS
    assert loaded_model.recommend(user_id=2, k=2) == expected


def test_save_creates_parent_directory(
    interactions: pd.DataFrame,
    tmp_path: Path,
) -> None:
    model_path = tmp_path / "nested" / "item_knn.pkl"
    model = ItemKNNRecommender().fit(interactions)

    model.save(model_path)

    assert model_path.exists()


def test_save_rejects_unfitted_model(tmp_path: Path) -> None:
    model = ItemKNNRecommender()
    model_path = tmp_path / "item_knn.pkl"

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
        ItemKNNRecommender.load(model_path)


def test_load_rejects_invalid_payload(tmp_path: Path) -> None:
    model_path = tmp_path / "invalid.pkl"

    with model_path.open("wb") as model_file:
        pickle.dump(["invalid"], model_file)

    with pytest.raises(
        ValueError,
        match="Invalid Item-KNN model payload",
    ):
        ItemKNNRecommender.load(model_path)
