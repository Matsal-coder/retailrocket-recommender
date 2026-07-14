"""Base interfaces for event preprocessing strategies."""

from abc import ABC, abstractmethod

import pandas as pd


class EventPreprocessor(ABC):
    """Define the contract for event preprocessing strategies."""

    @abstractmethod
    def transform(self, events: pd.DataFrame) -> pd.DataFrame:
        """Transform raw events into a standardized event dataset.

        Args:
            events: Raw event data.

        Returns:
            A standardized DataFrame ready for feature engineering.
        """
