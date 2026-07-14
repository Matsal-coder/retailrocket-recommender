"""Implicit-feedback preprocessing strategy."""

from collections.abc import Mapping, Sequence

import pandas as pd

from retail_recommender.data.preprocessors.base import EventPreprocessor

RAW_COLUMN_MAPPING = {
    "visitorid": "user_id",
    "itemid": "item_id",
    "event": "event_type",
}

REQUIRED_COLUMNS = {
    "timestamp",
    "user_id",
    "item_id",
    "event_type",
}

OUTPUT_COLUMNS = [
    "user_id",
    "item_id",
    "event_type",
    "event_weight",
    "timestamp",
    "datetime",
]


class ImplicitFeedbackPreprocessor(EventPreprocessor):
    """Convert RetailRocket events into weighted implicit feedback."""

    def __init__(
        self,
        event_weights: Mapping[str, float],
        allowed_event_types: Sequence[str] | None = None,
    ) -> None:
        """Initialize the implicit-feedback preprocessor.

        Args:
            event_weights: Weight assigned to each supported event type.
            allowed_event_types: Event types that may remain in the dataset.

        Raises:
            ValueError: If no event weights are provided or an allowed event
                does not have a configured weight.
        """
        normalized_weights = {
            str(event_type).strip().lower(): float(weight)
            for event_type, weight in event_weights.items()
        }

        if not normalized_weights:
            msg = "At least one event weight must be configured."
            raise ValueError(msg)

        normalized_allowed_events = (
            tuple(str(event).strip().lower() for event in allowed_event_types)
            if allowed_event_types is not None
            else tuple(normalized_weights)
        )

        missing_weights = set(normalized_allowed_events) - set(normalized_weights)
        if missing_weights:
            missing = ", ".join(sorted(missing_weights))
            msg = f"Missing weights for allowed event types: {missing}."
            raise ValueError(msg)

        self._event_weights = normalized_weights
        self._allowed_event_types = frozenset(normalized_allowed_events)

    def transform(self, events: pd.DataFrame) -> pd.DataFrame:
        """Clean raw events and map them to weighted implicit feedback.

        Args:
            events: Raw RetailRocket events.

        Returns:
            DataFrame with standardized IDs, event types, weights and dates.

        Raises:
            ValueError: If required columns are missing.
        """
        standardized_events = self._standardize_column_names(events)
        self._validate_required_columns(standardized_events)

        cleaned_events = standardized_events.loc[
            :, ["timestamp", "user_id", "item_id", "event_type"]
        ].copy()

        cleaned_events["event_type"] = (
            cleaned_events["event_type"].astype("string").str.strip().str.lower()
        )
        cleaned_events["timestamp"] = pd.to_numeric(
            cleaned_events["timestamp"],
            errors="coerce",
        )
        cleaned_events["user_id"] = pd.to_numeric(
            cleaned_events["user_id"],
            errors="coerce",
        )
        cleaned_events["item_id"] = pd.to_numeric(
            cleaned_events["item_id"],
            errors="coerce",
        )

        cleaned_events = cleaned_events.dropna(
            subset=["timestamp", "user_id", "item_id", "event_type"],
        )

        cleaned_events = cleaned_events[
            cleaned_events["event_type"].isin(self._allowed_event_types)
        ].copy()

        cleaned_events["user_id"] = cleaned_events["user_id"].astype("int64")
        cleaned_events["item_id"] = cleaned_events["item_id"].astype("int64")
        cleaned_events["timestamp"] = cleaned_events["timestamp"].astype("int64")

        cleaned_events["event_weight"] = (
            cleaned_events["event_type"].map(self._event_weights).astype("float64")
        )
        cleaned_events["datetime"] = pd.to_datetime(
            cleaned_events["timestamp"],
            unit="ms",
            utc=True,
        )

        cleaned_events = (
            cleaned_events.loc[:, OUTPUT_COLUMNS]
            .sort_values(["timestamp", "user_id", "item_id"])
            .reset_index(drop=True)
        )

        return cleaned_events

    @staticmethod
    def _standardize_column_names(events: pd.DataFrame) -> pd.DataFrame:
        """Normalize column names used by the raw RetailRocket dataset."""
        normalized_events = events.copy()
        normalized_events.columns = [
            str(column).strip().lower() for column in normalized_events.columns
        ]

        return normalized_events.rename(columns=RAW_COLUMN_MAPPING)

    @staticmethod
    def _validate_required_columns(events: pd.DataFrame) -> None:
        """Validate that the standardized dataset has all required columns."""
        missing_columns = REQUIRED_COLUMNS - set(events.columns)

        if missing_columns:
            missing = ", ".join(sorted(missing_columns))
            msg = f"Missing required event columns: {missing}."
            raise ValueError(msg)
