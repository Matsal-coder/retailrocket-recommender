"""Build aggregated user-item interactions from clean events."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

REQUIRED_EVENT_COLUMNS = {
    "user_id",
    "item_id",
    "event_weight",
    "datetime",
}

INTERACTION_COLUMNS = [
    "user_id",
    "item_id",
    "interaction_score",
    "interaction_count",
    "last_interaction_at",
    "target",
]

POSITIVE_TARGET = 1


@dataclass(frozen=True)
class InteractionFilterConfig:
    """Configure minimum activity requirements for interactions."""

    minimum_user_interactions: int = 1
    minimum_item_interactions: int = 1

    def __post_init__(self) -> None:
        """Validate minimum interaction thresholds."""
        if self.minimum_user_interactions < 1:
            msg = "minimum_user_interactions must be at least 1."
            raise ValueError(msg)

        if self.minimum_item_interactions < 1:
            msg = "minimum_item_interactions must be at least 1."
            raise ValueError(msg)


def build_interactions(events: pd.DataFrame) -> pd.DataFrame:
    """Aggregate clean events into positive user-item interactions.

    Args:
        events: Clean event-level data.

    Returns:
        Aggregated positive interactions, one row per user-item pair.

    Raises:
        ValueError: If required columns are absent or the input is empty.
    """
    _validate_events(events)

    interactions = (
        events.groupby(
            ["user_id", "item_id"],
            as_index=False,
            sort=False,
        )
        .agg(
            interaction_score=("event_weight", "sum"),
            interaction_count=("event_weight", "size"),
            last_interaction_at=("datetime", "max"),
        )
        .assign(target=POSITIVE_TARGET)
    )

    interactions["interaction_score"] = interactions["interaction_score"].astype(
        "float64"
    )
    interactions["interaction_count"] = interactions["interaction_count"].astype(
        "int64"
    )
    interactions["target"] = interactions["target"].astype("int8")

    return (
        interactions.loc[:, INTERACTION_COLUMNS]
        .sort_values(
            ["last_interaction_at", "user_id", "item_id"],
        )
        .reset_index(drop=True)
    )


def filter_interactions(
    interactions: pd.DataFrame,
    config: InteractionFilterConfig,
) -> pd.DataFrame:
    """Filter users and items that do not meet minimum activity levels.

    Filtering is repeated until all remaining users and items satisfy the
    configured thresholds.

    Args:
        interactions: Aggregated user-item interactions.
        config: Minimum user and item activity thresholds.

    Returns:
        Filtered interactions.

    Raises:
        ValueError: If required interaction columns are absent.
    """
    _validate_interactions(interactions)

    filtered = interactions.copy()

    while not filtered.empty:
        previous_size = len(filtered)

        user_counts = filtered.groupby("user_id")["item_id"].transform("nunique")
        filtered = filtered[user_counts >= config.minimum_user_interactions].copy()

        item_counts = filtered.groupby("item_id")["user_id"].transform("nunique")
        filtered = filtered[item_counts >= config.minimum_item_interactions].copy()

        if len(filtered) == previous_size:
            break

    return filtered.reset_index(drop=True)


def build_and_filter_interactions(
    events: pd.DataFrame,
    config: InteractionFilterConfig,
) -> pd.DataFrame:
    """Build aggregated interactions and apply activity filters.

    Args:
        events: Clean event-level data.
        config: Minimum user and item activity thresholds.

    Returns:
        Aggregated and filtered positive interactions.
    """
    interactions = build_interactions(events)
    return filter_interactions(interactions, config)


def _validate_events(events: pd.DataFrame) -> None:
    """Validate clean events before aggregation."""
    if events.empty:
        msg = "Cannot build interactions from an empty event dataset."
        raise ValueError(msg)

    missing_columns = REQUIRED_EVENT_COLUMNS - set(events.columns)

    if missing_columns:
        missing = ", ".join(sorted(missing_columns))
        msg = f"Missing required event columns: {missing}."
        raise ValueError(msg)


def _validate_interactions(interactions: pd.DataFrame) -> None:
    """Validate aggregated interactions before filtering."""
    required_columns = {"user_id", "item_id"}
    missing_columns = required_columns - set(interactions.columns)

    if missing_columns:
        missing = ", ".join(sorted(missing_columns))
        msg = f"Missing required interaction columns: {missing}."
        raise ValueError(msg)
