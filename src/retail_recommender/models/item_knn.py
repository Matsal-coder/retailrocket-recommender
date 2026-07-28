"""Item-based K-Nearest Neighbors recommendation baseline."""

from __future__ import annotations

import pickle
from collections.abc import Collection
from pathlib import Path
from typing import Any

import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.neighbors import NearestNeighbors

from retail_recommender.models.base import BaseRecommender

REQUIRED_COLUMNS = {
    "user_idx",
    "item_idx",
    "interaction_score",
}


class ItemKNNRecommender(BaseRecommender[int, int]):
    """Recommend items using item-to-item cosine similarity."""

    def __init__(
        self,
        n_neighbors: int = 20,
        minimum_similarity: float = 0.0,
    ) -> None:
        """Initialize an unfitted Item-KNN recommender.

        Args:
            n_neighbors: Maximum number of similar items stored per item.
            minimum_similarity: Minimum cosine similarity accepted when
                producing recommendations.

        Raises:
            ValueError: If configuration values are invalid.
        """
        if n_neighbors <= 0:
            msg = "n_neighbors must be greater than zero"
            raise ValueError(msg)

        if not 0.0 <= minimum_similarity <= 1.0:
            msg = "minimum_similarity must be between zero and one"
            raise ValueError(msg)

        self.n_neighbors = n_neighbors
        self.minimum_similarity = minimum_similarity

        self._user_items: dict[int, dict[int, float]] = {}
        self._item_neighbors: dict[int, list[tuple[int, float]]] = {}
        self._popular_items: list[int] = []
        self._is_fitted = False

    @property
    def is_fitted(self) -> bool:
        """Return whether the recommender has been fitted."""
        return self._is_fitted

    @property
    def item_neighbors(self) -> dict[int, list[tuple[int, float]]]:
        """Return a copy of the learned item-neighbor mapping."""
        return {
            item_id: neighbors.copy()
            for item_id, neighbors in self._item_neighbors.items()
        }

    def fit(
        self,
        interactions: Any,
    ) -> ItemKNNRecommender:
        """Fit the recommender from user-item interactions.

        Args:
            interactions: DataFrame containing user IDs, item IDs and
                interaction scores.

        Returns:
            The fitted recommender instance.

        Raises:
            TypeError: If interactions is not a DataFrame.
            ValueError: If data is empty or required columns are missing.
        """
        if not isinstance(interactions, pd.DataFrame):
            msg = "interactions must be a pandas DataFrame"
            raise TypeError(msg)

        self._validate_columns(interactions)

        if interactions.empty:
            msg = "interactions must not be empty"
            raise ValueError(msg)

        aggregated = (
            interactions.groupby(
                ["user_idx", "item_idx"],
                as_index=False,
            )["interaction_score"]
            .sum()
            .sort_values(
                ["user_idx", "item_idx"],
                kind="stable",
            )
        )

        user_ids = sorted(aggregated["user_idx"].astype(int).unique().tolist())
        item_ids = sorted(aggregated["item_idx"].astype(int).unique().tolist())

        user_to_position = {
            user_id: position for position, user_id in enumerate(user_ids)
        }
        item_to_position = {
            item_id: position for position, item_id in enumerate(item_ids)
        }

        row_positions = (
            aggregated["user_idx"].astype(int).map(user_to_position).to_numpy()
        )
        column_positions = (
            aggregated["item_idx"].astype(int).map(item_to_position).to_numpy()
        )
        scores = aggregated["interaction_score"].astype(float).to_numpy()

        user_item_matrix = csr_matrix(
            (
                scores,
                (row_positions, column_positions),
            ),
            shape=(len(user_ids), len(item_ids)),
        )

        self._user_items = self._build_user_history(aggregated)
        self._popular_items = self._build_popularity_ranking(aggregated)
        self._item_neighbors = self._fit_item_neighbors(
            item_ids=item_ids,
            user_item_matrix=user_item_matrix,
        )
        self._is_fitted = True

        return self

    def recommend(
        self,
        user_id: int,
        k: int,
        seen_items: Collection[int] | None = None,
    ) -> list[int]:
        """Recommend Top-K unseen items for a user.

        Args:
            user_id: Encoded user identifier.
            k: Maximum number of recommendations.
            seen_items: Optional additional items to exclude.

        Returns:
            Ranked unseen item identifiers.

        Raises:
            RuntimeError: If the model has not been fitted.
            ValueError: If K is not greater than zero.
        """
        self._ensure_fitted()
        self._validate_k(k)

        user_history = self._user_items.get(user_id, {})
        excluded_items = set(user_history)
        excluded_items.update(seen_items or [])

        candidate_scores: dict[int, float] = {}

        for source_item, interaction_score in user_history.items():
            neighbors = self._item_neighbors.get(source_item, [])

            for neighbor_item, similarity in neighbors:
                if neighbor_item in excluded_items:
                    continue

                weighted_score = interaction_score * similarity
                candidate_scores[neighbor_item] = (
                    candidate_scores.get(neighbor_item, 0.0) + weighted_score
                )

        ranked_candidates = sorted(
            candidate_scores,
            key=lambda item_id: (
                -candidate_scores[item_id],
                item_id,
            ),
        )

        recommendations = ranked_candidates[:k]

        if len(recommendations) < k:
            recommendations.extend(
                self._popularity_fallback(
                    already_recommended=set(recommendations),
                    excluded_items=excluded_items,
                    limit=k - len(recommendations),
                )
            )

        return recommendations

    def save(self, path: Path) -> None:
        """Serialize the fitted recommender.

        Args:
            path: Destination path.

        Raises:
            RuntimeError: If the model has not been fitted.
        """
        self._ensure_fitted()

        path.parent.mkdir(parents=True, exist_ok=True)

        payload = {
            "n_neighbors": self.n_neighbors,
            "minimum_similarity": self.minimum_similarity,
            "user_items": self._user_items,
            "item_neighbors": self._item_neighbors,
            "popular_items": self._popular_items,
            "is_fitted": self._is_fitted,
        }

        with path.open("wb") as model_file:
            pickle.dump(payload, model_file)

    @classmethod
    def load(cls, path: Path) -> ItemKNNRecommender:
        """Load a serialized Item-KNN recommender.

        Args:
            path: Serialized model path.

        Returns:
            Restored recommender.

        Raises:
            FileNotFoundError: If the file does not exist.
            ValueError: If the payload is invalid.
        """
        if not path.exists():
            msg = f"Model file not found: {path}"
            raise FileNotFoundError(msg)

        with path.open("rb") as model_file:
            payload = pickle.load(model_file)

        required_keys = {
            "n_neighbors",
            "minimum_similarity",
            "user_items",
            "item_neighbors",
            "popular_items",
            "is_fitted",
        }

        if (
            not isinstance(payload, dict)
            or not required_keys.issubset(payload)
            or payload["is_fitted"] is not True
        ):
            msg = "Invalid Item-KNN model payload"
            raise ValueError(msg)

        model = cls(
            n_neighbors=int(payload["n_neighbors"]),
            minimum_similarity=float(payload["minimum_similarity"]),
        )
        model._user_items = payload["user_items"]
        model._item_neighbors = payload["item_neighbors"]
        model._popular_items = payload["popular_items"]
        model._is_fitted = True

        return model

    def _fit_item_neighbors(
        self,
        item_ids: list[int],
        user_item_matrix: csr_matrix,
    ) -> dict[int, list[tuple[int, float]]]:
        """Fit nearest neighbors and map similar items."""
        item_user_matrix = user_item_matrix.transpose().tocsr()

        neighbor_count = min(
            self.n_neighbors + 1,
            len(item_ids),
        )

        nearest_neighbors = NearestNeighbors(
            metric="cosine",
            algorithm="brute",
            n_neighbors=neighbor_count,
        )
        nearest_neighbors.fit(item_user_matrix)

        distances, indices = nearest_neighbors.kneighbors(item_user_matrix)

        item_neighbors: dict[int, list[tuple[int, float]]] = {}

        for item_position, item_id in enumerate(item_ids):
            neighbors: list[tuple[int, float]] = []

            for neighbor_position, distance in zip(
                indices[item_position],
                distances[item_position],
                strict=True,
            ):
                neighbor_item = item_ids[int(neighbor_position)]

                if neighbor_item == item_id:
                    continue

                similarity = 1.0 - float(distance)

                if similarity < self.minimum_similarity:
                    continue

                neighbors.append((neighbor_item, similarity))

                if len(neighbors) == self.n_neighbors:
                    break

            item_neighbors[item_id] = neighbors

        return item_neighbors

    @staticmethod
    def _build_user_history(
        interactions: pd.DataFrame,
    ) -> dict[int, dict[int, float]]:
        """Build weighted interaction histories grouped by user."""
        user_items: dict[int, dict[int, float]] = {}

        for row in interactions.itertuples(index=False):
            user_id = int(row.user_idx)
            item_id = int(row.item_idx)
            interaction_score = float(row.interaction_score)

            user_items.setdefault(user_id, {})[item_id] = interaction_score

        return user_items

    @staticmethod
    def _build_popularity_ranking(
        interactions: pd.DataFrame,
    ) -> list[int]:
        """Build deterministic popularity fallback ranking."""
        popularity = (
            interactions.groupby("item_idx", as_index=False)
            .agg(
                popularity_score=("interaction_score", "sum"),
                interaction_count=("user_idx", "count"),
            )
            .sort_values(
                [
                    "popularity_score",
                    "interaction_count",
                    "item_idx",
                ],
                ascending=[False, False, True],
                kind="stable",
            )
        )

        return popularity["item_idx"].astype(int).tolist()

    def _popularity_fallback(
        self,
        already_recommended: set[int],
        excluded_items: set[int],
        limit: int,
    ) -> list[int]:
        """Fill missing recommendation slots with popular unseen items."""
        fallback_items: list[int] = []

        for item_id in self._popular_items:
            if item_id in already_recommended:
                continue

            if item_id in excluded_items:
                continue

            fallback_items.append(item_id)

            if len(fallback_items) == limit:
                break

        return fallback_items

    @staticmethod
    def _validate_columns(interactions: pd.DataFrame) -> None:
        """Validate required input columns."""
        missing_columns = REQUIRED_COLUMNS.difference(interactions.columns)

        if missing_columns:
            formatted_columns = ", ".join(sorted(missing_columns))
            msg = "Missing required interaction columns: " f"{formatted_columns}"
            raise ValueError(msg)

    @staticmethod
    def _validate_k(k: int) -> None:
        """Validate recommendation cutoff."""
        if k <= 0:
            msg = "k must be greater than zero"
            raise ValueError(msg)

    def _ensure_fitted(self) -> None:
        """Ensure that the model has been fitted."""
        if not self._is_fitted:
            msg = "ItemKNNRecommender must be fitted before use"
            raise RuntimeError(msg)
