"""Popularity-based recommendation baseline."""

from __future__ import annotations

import pickle
from collections.abc import Collection
from pathlib import Path
from typing import Any

import pandas as pd

from retail_recommender.models.base import BaseRecommender

REQUIRED_COLUMNS = {
    "item_idx",
    "interaction_score",
    "interaction_count",
}


class PopularityRecommender(BaseRecommender[int, int]):
    """Recommend globally popular items learned from training interactions."""

    def __init__(self) -> None:
        """Initialize an unfitted popularity recommender."""
        self._ranked_items: list[int] = []
        self._is_fitted = False

    @property
    def is_fitted(self) -> bool:
        """Return whether the recommender has already been fitted."""
        return self._is_fitted

    @property
    def ranked_items(self) -> list[int]:
        """Return a copy of items ordered by global popularity."""
        return self._ranked_items.copy()

    def fit(
        self,
        interactions: Any,
    ) -> PopularityRecommender:
        """Fit the recommender from user-item interaction data.

        Args:
            interactions: Pandas DataFrame containing item identifiers,
                interaction scores and interaction counts.

        Returns:
            The fitted recommender instance.

        Raises:
            TypeError: If interactions is not a pandas DataFrame.
            ValueError: If required columns are missing or data is empty.
        """
        if not isinstance(interactions, pd.DataFrame):
            msg = "interactions must be a pandas DataFrame"
            raise TypeError(msg)

        self._validate_columns(interactions)

        if interactions.empty:
            msg = "interactions must not be empty"
            raise ValueError(msg)

        popularity = (
            interactions.groupby("item_idx", as_index=False)
            .agg(
                popularity_score=("interaction_score", "sum"),
                popularity_count=("interaction_count", "sum"),
            )
            .sort_values(
                by=[
                    "popularity_score",
                    "popularity_count",
                    "item_idx",
                ],
                ascending=[False, False, True],
                kind="stable",
            )
        )

        self._ranked_items = popularity["item_idx"].astype(int).tolist()
        self._is_fitted = True

        return self

    def recommend(
        self,
        user_id: int,
        k: int,
        seen_items: Collection[int] | None = None,
    ) -> list[int]:
        """Recommend globally popular unseen items.

        Args:
            user_id: User identifier. It does not alter global popularity,
                but remains part of the shared recommender interface.
            k: Maximum number of items to recommend.
            seen_items: Items that should be excluded.

        Returns:
            Ranked unseen item identifiers.

        Raises:
            RuntimeError: If the model has not been fitted.
            ValueError: If K is not greater than zero.
        """
        del user_id

        self._ensure_fitted()
        self._validate_k(k)

        excluded_items = set(seen_items or [])

        recommendations = [
            item for item in self._ranked_items if item not in excluded_items
        ]

        return recommendations[:k]

    def save(self, path: Path) -> None:
        """Serialize the fitted recommender to disk.

        Args:
            path: Destination path for the serialized model.

        Raises:
            RuntimeError: If the model has not been fitted.
        """
        self._ensure_fitted()

        path.parent.mkdir(parents=True, exist_ok=True)

        payload = {
            "ranked_items": self._ranked_items,
            "is_fitted": self._is_fitted,
        }

        with path.open("wb") as model_file:
            pickle.dump(payload, model_file)

    @classmethod
    def load(cls, path: Path) -> PopularityRecommender:
        """Load a serialized popularity recommender.

        Args:
            path: Path containing the serialized model.

        Returns:
            Restored popularity recommender.

        Raises:
            FileNotFoundError: If the model file does not exist.
            ValueError: If the serialized payload is invalid.
        """
        if not path.exists():
            msg = f"Model file not found: {path}"
            raise FileNotFoundError(msg)

        with path.open("rb") as model_file:
            payload = pickle.load(model_file)

        if not isinstance(payload, dict):
            msg = "Invalid popularity model payload"
            raise ValueError(msg)

        ranked_items = payload.get("ranked_items")
        is_fitted = payload.get("is_fitted")

        if not isinstance(ranked_items, list) or is_fitted is not True:
            msg = "Invalid popularity model payload"
            raise ValueError(msg)

        model = cls()
        model._ranked_items = [int(item) for item in ranked_items]
        model._is_fitted = True

        return model

    @staticmethod
    def _validate_columns(interactions: pd.DataFrame) -> None:
        """Validate required interaction columns."""
        missing_columns = REQUIRED_COLUMNS.difference(interactions.columns)

        if missing_columns:
            formatted_columns = ", ".join(sorted(missing_columns))
            msg = "Missing required interaction columns: " f"{formatted_columns}"
            raise ValueError(msg)

    @staticmethod
    def _validate_k(k: int) -> None:
        """Validate the recommendation cutoff."""
        if k <= 0:
            msg = "k must be greater than zero"
            raise ValueError(msg)

    def _ensure_fitted(self) -> None:
        """Ensure that the recommender has been fitted."""
        if not self._is_fitted:
            msg = "PopularityRecommender must be fitted before use"
            raise RuntimeError(msg)
