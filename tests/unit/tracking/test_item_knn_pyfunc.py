"""Unit tests for the Item-KNN MLflow PyFunc wrapper."""

from types import SimpleNamespace
from unittest.mock import MagicMock

import pandas as pd
import pytest

from retail_recommender.tracking import item_knn_pyfunc
from retail_recommender.tracking.item_knn_pyfunc import (
    ItemKNNPyfuncModel,
)

DEFAULT_K = 2


def test_pyfunc_rejects_invalid_configuration() -> None:
    with pytest.raises(ValueError, match="n_neighbors"):
        ItemKNNPyfuncModel(
            n_neighbors=0,
            minimum_similarity=0.0,
            default_k=DEFAULT_K,
        )

    with pytest.raises(ValueError, match="default_k"):
        ItemKNNPyfuncModel(
            n_neighbors=2,
            minimum_similarity=0.0,
            default_k=0,
        )


def test_load_context_fits_recommender(
    tmp_path,
    monkeypatch,
) -> None:
    train_path = tmp_path / "train.parquet"
    pd.DataFrame(
        {
            "user_idx": [0, 0, 1],
            "item_idx": [0, 1, 2],
            "interaction_score": [1.0, 2.0, 1.0],
            "interaction_count": [1, 1, 1],
            "target": [1, 1, 1],
        }
    ).to_parquet(train_path, index=False)

    recommender = MagicMock()
    constructor = MagicMock(return_value=recommender)
    monkeypatch.setattr(
        item_knn_pyfunc,
        "ItemKNNRecommender",
        constructor,
    )

    model = ItemKNNPyfuncModel(
        n_neighbors=2,
        minimum_similarity=0.0,
        default_k=DEFAULT_K,
    )
    context = SimpleNamespace(
        artifacts={
            "train_interactions": str(train_path),
        }
    )

    model.load_context(context)

    constructor.assert_called_once_with(
        n_neighbors=2,
        minimum_similarity=0.0,
    )
    recommender.fit.assert_called_once()


def test_predict_returns_encoded_recommendations() -> None:
    model = ItemKNNPyfuncModel(
        n_neighbors=2,
        minimum_similarity=0.0,
        default_k=DEFAULT_K,
    )
    recommender = MagicMock()
    recommender.recommend.return_value = [3, 4]
    model._recommender = recommender
    model._seen_items_by_user = {0: {1, 2}}

    result = model.predict(
        context=MagicMock(),
        model_input=pd.DataFrame({"user_idx": [0]}),
    )

    assert result.to_dict(orient="records") == [
        {
            "user_idx": 0,
            "recommendations": "[3, 4]",
        }
    ]
    recommender.recommend.assert_called_once_with(
        user_id=0,
        k=DEFAULT_K,
        seen_items={1, 2},
    )


def test_predict_rejects_missing_column() -> None:
    model = ItemKNNPyfuncModel(
        n_neighbors=2,
        minimum_similarity=0.0,
        default_k=DEFAULT_K,
    )
    model._recommender = MagicMock()

    with pytest.raises(ValueError, match="user_idx"):
        model.predict(
            context=MagicMock(),
            model_input=pd.DataFrame({"invalid": [0]}),
        )


def test_predict_rejects_unloaded_model() -> None:
    model = ItemKNNPyfuncModel(
        n_neighbors=2,
        minimum_similarity=0.0,
        default_k=DEFAULT_K,
    )

    with pytest.raises(RuntimeError, match="has not been loaded"):
        model.predict(
            context=MagicMock(),
            model_input=pd.DataFrame({"user_idx": [0]}),
        )
