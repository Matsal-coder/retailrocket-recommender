"""Factory for recommender model creation."""

from __future__ import annotations

from typing import Any

from retail_recommender.models.base import BaseRecommender
from retail_recommender.models.item_knn import ItemKNNRecommender
from retail_recommender.models.popularity import PopularityRecommender

POPULARITY_MODEL_NAME = "popularity"
ITEM_KNN_MODEL_NAME = "item_knn"

SUPPORTED_MODELS = {
    POPULARITY_MODEL_NAME,
    ITEM_KNN_MODEL_NAME,
}


def create_recommender(
    model_name: str,
    **model_params: Any,
) -> BaseRecommender[int, int]:
    """Create a recommender from its configured name.

    Args:
        model_name: Name identifying the recommender implementation.
        **model_params: Model-specific constructor parameters.

    Returns:
        Instantiated recommender model.

    Raises:
        TypeError: If model_name is not a string.
        ValueError: If the normalized model name is empty or unsupported.
        TypeError: If unsupported constructor parameters are provided.
    """
    normalized_name = normalize_model_name(model_name)

    if normalized_name == POPULARITY_MODEL_NAME:
        return _create_popularity_recommender(model_params)

    if normalized_name == ITEM_KNN_MODEL_NAME:
        return _create_item_knn_recommender(model_params)

    supported_models = ", ".join(sorted(SUPPORTED_MODELS))
    msg = (
        f"Unsupported recommender model: {model_name}. "
        f"Supported models: {supported_models}"
    )
    raise ValueError(msg)


def normalize_model_name(model_name: str) -> str:
    """Normalize a recommender name for factory lookup.

    Args:
        model_name: Raw recommender name.

    Returns:
        Normalized model name.

    Raises:
        TypeError: If model_name is not a string.
        ValueError: If model_name becomes empty after normalization.
    """
    if not isinstance(model_name, str):
        msg = "model_name must be a string"
        raise TypeError(msg)

    normalized_name = model_name.strip().lower().replace("-", "_").replace(" ", "_")

    if not normalized_name:
        msg = "model_name must not be empty"
        raise ValueError(msg)

    return normalized_name


def _create_popularity_recommender(
    model_params: dict[str, Any],
) -> PopularityRecommender:
    """Create a popularity recommender.

    The popularity baseline currently does not accept configuration
    parameters.

    Args:
        model_params: Constructor parameters supplied to the factory.

    Returns:
        Popularity recommender instance.

    Raises:
        TypeError: If unexpected parameters are supplied.
    """
    if model_params:
        unexpected_params = ", ".join(sorted(model_params))
        msg = (
            "PopularityRecommender does not accept parameters: " f"{unexpected_params}"
        )
        raise TypeError(msg)

    return PopularityRecommender()


def _create_item_knn_recommender(
    model_params: dict[str, Any],
) -> ItemKNNRecommender:
    """Create an Item-KNN recommender.

    Args:
        model_params: Item-KNN constructor parameters.

    Returns:
        Item-KNN recommender instance.
    """
    return ItemKNNRecommender(**model_params)
