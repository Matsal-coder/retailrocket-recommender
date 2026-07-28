"""Base interface for recommender models."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Collection, Hashable
from pathlib import Path
from typing import Any, Generic, TypeVar

UserId = TypeVar("UserId", bound=Hashable)
ItemId = TypeVar("ItemId", bound=Hashable)


class BaseRecommender(ABC, Generic[UserId, ItemId]):
    """Define the common interface for recommender models."""

    @abstractmethod
    def fit(self, interactions: Any) -> BaseRecommender[UserId, ItemId]:
        """Fit the recommender using interaction data.

        Args:
            interactions: Training interactions in the format required by the
                concrete recommender.

        Returns:
            The fitted recommender instance.
        """

    @abstractmethod
    def recommend(
        self,
        user_id: UserId,
        k: int,
        seen_items: Collection[ItemId] | None = None,
    ) -> list[ItemId]:
        """Recommend the Top-K items for a user.

        Args:
            user_id: Identifier of the user receiving recommendations.
            k: Maximum number of items to recommend.
            seen_items: Items that should be excluded from recommendations.

        Returns:
            Ranked item identifiers, ordered from most to least relevant.
        """

    @abstractmethod
    def save(self, path: Path) -> None:
        """Persist the fitted recommender.

        Args:
            path: Destination path for the serialized recommender.
        """

    @classmethod
    @abstractmethod
    def load(
        cls,
        path: Path,
    ) -> BaseRecommender[UserId, ItemId]:
        """Load a previously persisted recommender.

        Args:
            path: Path containing the serialized recommender.

        Returns:
            Restored recommender instance.
        """
