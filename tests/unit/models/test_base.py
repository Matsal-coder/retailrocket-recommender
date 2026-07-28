"""Unit tests for the recommender base interface."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from retail_recommender.models.base import BaseRecommender


class IncompleteRecommender(BaseRecommender[int, int]):
    """Recommender that intentionally omits abstract implementations."""


class DummyRecommender(BaseRecommender[int, int]):
    """Minimal implementation used to test the base contract."""

    def __init__(self) -> None:
        self.is_fitted = False

    def fit(self, interactions: Any) -> DummyRecommender:
        """Mark the dummy recommender as fitted."""
        self.is_fitted = interactions is not None
        return self

    def recommend(
        self,
        user_id: int,
        k: int,
        seen_items: set[int] | None = None,
    ) -> list[int]:
        """Return deterministic dummy recommendations."""
        del user_id

        excluded_items = seen_items or set()
        candidates = [10, 20, 30, 40]

        return [item for item in candidates if item not in excluded_items][:k]

    def save(self, path: Path) -> None:
        """Persist a marker file."""
        path.write_text("dummy", encoding="utf-8")

    @classmethod
    def load(cls, path: Path) -> DummyRecommender:
        """Restore the dummy recommender from a marker file."""
        if not path.exists():
            msg = f"Model file not found: {path}"
            raise FileNotFoundError(msg)

        model = cls()
        model.is_fitted = True
        return model


def test_base_recommender_cannot_be_instantiated_directly() -> None:
    with pytest.raises(TypeError):
        BaseRecommender()


def test_incomplete_recommender_cannot_be_instantiated() -> None:
    with pytest.raises(TypeError):
        IncompleteRecommender()


def test_complete_recommender_can_be_instantiated() -> None:
    model = DummyRecommender()

    assert isinstance(model, BaseRecommender)
    assert not model.is_fitted


def test_fit_returns_fitted_recommender() -> None:
    model = DummyRecommender()

    fitted_model = model.fit(interactions=[(1, 10)])

    assert fitted_model is model
    assert model.is_fitted


def test_recommend_returns_top_k_items() -> None:
    model = DummyRecommender()

    recommendations = model.recommend(user_id=1, k=2)

    assert recommendations == [10, 20]


def test_recommend_excludes_seen_items() -> None:
    model = DummyRecommender()

    recommendations = model.recommend(
        user_id=1,
        k=2,
        seen_items={10, 30},
    )

    assert recommendations == [20, 40]


def test_model_can_be_saved_and_loaded(tmp_path: Path) -> None:
    model_path = tmp_path / "dummy_model.txt"
    model = DummyRecommender()

    model.save(model_path)
    loaded_model = DummyRecommender.load(model_path)

    assert model_path.read_text(encoding="utf-8") == "dummy"
    assert loaded_model.is_fitted


def test_load_rejects_missing_model_file(tmp_path: Path) -> None:
    model_path = tmp_path / "missing_model.txt"

    with pytest.raises(
        FileNotFoundError,
        match="Model file not found",
    ):
        DummyRecommender.load(model_path)
