"""MLflow PyFunc wrapper for the Item-KNN recommender."""

from __future__ import annotations

import json
from typing import Any

import mlflow.pyfunc
import pandas as pd

from retail_recommender.models.item_knn import ItemKNNRecommender

TRAIN_INTERACTIONS_ARTIFACT = "train_interactions"
REQUIRED_INPUT_COLUMNS = {"user_idx"}


class ItemKNNPyfuncModel(mlflow.pyfunc.PythonModel):
    """Expose Item-KNN recommendations through the MLflow PyFunc interface."""

    def __init__(
        self,
        *,
        n_neighbors: int,
        minimum_similarity: float,
        default_k: int,
    ) -> None:
        """Store model configuration before MLflow serialization."""

        if n_neighbors <= 0:
            raise ValueError("n_neighbors must be greater than zero.")

        if default_k <= 0:
            raise ValueError("default_k must be greater than zero.")

        self._n_neighbors = n_neighbors
        self._minimum_similarity = minimum_similarity
        self._default_k = default_k
        self._recommender: ItemKNNRecommender | None = None
        self._seen_items_by_user: dict[int, set[int]] = {}

    def load_context(
        self,
        context: mlflow.pyfunc.PythonModelContext,
    ) -> None:
        """Load training interactions and rebuild the fitted Item-KNN."""

        train_path = context.artifacts.get(
            TRAIN_INTERACTIONS_ARTIFACT,
        )

        if train_path is None:
            message = (
                "MLflow model is missing the "
                f"'{TRAIN_INTERACTIONS_ARTIFACT}' artifact."
            )
            raise ValueError(message)

        train_interactions = pd.read_parquet(train_path)

        self._recommender = ItemKNNRecommender(
            n_neighbors=self._n_neighbors,
            minimum_similarity=self._minimum_similarity,
        )
        self._recommender.fit(train_interactions)

        self._seen_items_by_user = (
            train_interactions.groupby("user_idx")["item_idx"]
            .apply(lambda values: {int(value) for value in values})
            .to_dict()
        )

    def predict(
        self,
        context: mlflow.pyfunc.PythonModelContext,
        model_input: pd.DataFrame,
        params: dict[str, Any] | None = None,
    ) -> pd.DataFrame:
        """Generate JSON-encoded Top-K recommendations for each input user."""

        del context

        if self._recommender is None:
            raise RuntimeError("Item-KNN model has not been loaded.")

        missing_columns = sorted(REQUIRED_INPUT_COLUMNS.difference(model_input.columns))

        if missing_columns:
            missing = ", ".join(missing_columns)
            message = f"Model input is missing columns: {missing}"
            raise ValueError(message)

        configured_k = (
            params.get("k", self._default_k) if params is not None else self._default_k
        )
        k = int(configured_k)

        if k <= 0:
            raise ValueError("Prediction parameter 'k' must be greater than zero.")

        rows: list[dict[str, Any]] = []

        for user_idx in model_input["user_idx"]:
            normalized_user_idx = int(user_idx)
            seen_items = self._seen_items_by_user.get(
                normalized_user_idx,
                set(),
            )
            recommendations = self._recommender.recommend(
                user_id=normalized_user_idx,
                k=k,
                seen_items=seen_items,
            )

            rows.append(
                {
                    "user_idx": normalized_user_idx,
                    "recommendations": json.dumps(
                        [int(item) for item in recommendations]
                    ),
                }
            )

        return pd.DataFrame(rows)
