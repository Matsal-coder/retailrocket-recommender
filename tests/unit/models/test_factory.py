"""Unit tests for the recommender model factory."""

from __future__ import annotations

from typing import Any

import pytest

from retail_recommender.models.factory import (
    create_recommender,
    normalize_model_name,
)
from retail_recommender.models.item_knn import ItemKNNRecommender
from retail_recommender.models.popularity import PopularityRecommender

DEFAULT_NEIGHBOR_COUNT = 25
DEFAULT_MINIMUM_SIMILARITY = 0.2


@pytest.mark.parametrize(
    ("raw_name", "expected_name"),
    [
        ("popularity", "popularity"),
        (" Popularity ", "popularity"),
        ("POPULARITY", "popularity"),
        ("item_knn", "item_knn"),
        ("item-knn", "item_knn"),
        ("item knn", "item_knn"),
        (" ITEM KNN ", "item_knn"),
    ],
)
def test_normalize_model_name(
    raw_name: str,
    expected_name: str,
) -> None:
    normalized_name = normalize_model_name(raw_name)

    assert normalized_name == expected_name


def test_normalize_model_name_rejects_non_string() -> None:
    invalid_name: Any = 123

    with pytest.raises(
        TypeError,
        match="model_name must be a string",
    ):
        normalize_model_name(invalid_name)


@pytest.mark.parametrize(
    "model_name",
    [
        "",
        " ",
        "   ",
    ],
)
def test_normalize_model_name_rejects_empty_name(
    model_name: str,
) -> None:
    with pytest.raises(
        ValueError,
        match="model_name must not be empty",
    ):
        normalize_model_name(model_name)


def test_create_recommender_returns_popularity_model() -> None:
    model = create_recommender("popularity")

    assert isinstance(model, PopularityRecommender)


@pytest.mark.parametrize(
    "model_name",
    [
        "item_knn",
        "item-knn",
        "ITEM KNN",
    ],
)
def test_create_recommender_returns_item_knn_model(
    model_name: str,
) -> None:
    model = create_recommender(model_name)

    assert isinstance(model, ItemKNNRecommender)


def test_create_recommender_passes_item_knn_parameters() -> None:
    model = create_recommender(
        "item_knn",
        n_neighbors=DEFAULT_NEIGHBOR_COUNT,
        minimum_similarity=DEFAULT_MINIMUM_SIMILARITY,
    )

    assert isinstance(model, ItemKNNRecommender)
    assert model.n_neighbors == DEFAULT_NEIGHBOR_COUNT
    assert model.minimum_similarity == DEFAULT_MINIMUM_SIMILARITY


def test_create_recommender_rejects_unknown_model() -> None:
    with pytest.raises(
        ValueError,
        match="Unsupported recommender model",
    ):
        create_recommender("unknown_model")


def test_unknown_model_error_lists_supported_models() -> None:
    with pytest.raises(ValueError) as error:
        create_recommender("matrix_factorization")

    error_message = str(error.value)

    assert "popularity" in error_message
    assert "item_knn" in error_message


def test_popularity_recommender_rejects_parameters() -> None:
    with pytest.raises(
        TypeError,
        match="does not accept parameters",
    ):
        create_recommender(
            "popularity",
            n_neighbors=DEFAULT_NEIGHBOR_COUNT,
        )


def test_item_knn_validation_is_preserved() -> None:
    with pytest.raises(
        ValueError,
        match="n_neighbors must be greater than zero",
    ):
        create_recommender(
            "item_knn",
            n_neighbors=0,
        )


def test_item_knn_rejects_unknown_constructor_parameter() -> None:
    with pytest.raises(TypeError):
        create_recommender(
            "item_knn",
            unsupported_parameter=True,
        )
