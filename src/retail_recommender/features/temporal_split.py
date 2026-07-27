"""Temporal splitting utilities for recommendation interactions."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd
from pandas.api.types import is_datetime64_any_dtype

REQUIRED_INTERACTION_COLUMNS = {
    "user_id",
    "item_id",
    "last_interaction_at",
}

TOTAL_SPLIT_SIZE = 1.0
SPLIT_TOLERANCE = 1e-9


@dataclass(frozen=True)
class TemporalSplitConfig:
    """Configure chronological train, validation and test proportions."""

    train_size: float
    validation_size: float
    test_size: float
    filter_unknown_entities: bool = True

    def __post_init__(self) -> None:
        """Validate temporal split proportions."""
        split_sizes = (
            self.train_size,
            self.validation_size,
            self.test_size,
        )

        if any(size <= 0 or size >= 1 for size in split_sizes):
            msg = "All split sizes must be greater than 0 and smaller than 1."
            raise ValueError(msg)

        total_size = sum(split_sizes)

        if abs(total_size - TOTAL_SPLIT_SIZE) > SPLIT_TOLERANCE:
            msg = "Train, validation and test sizes must sum to 1.0."
            raise ValueError(msg)


@dataclass(frozen=True)
class TemporalSplitResult:
    """Contain temporal data splits and cold-start filtering statistics."""

    train: pd.DataFrame
    validation: pd.DataFrame
    test: pd.DataFrame
    validation_removed_unknowns: int
    test_removed_unknowns: int


def temporal_split(
    interactions: pd.DataFrame,
    config: TemporalSplitConfig,
) -> TemporalSplitResult:
    """Split interactions chronologically into train, validation and test.

    Validation and test may be filtered to retain only users and items
    observed in the training set.

    Args:
        interactions: Aggregated positive user-item interactions.
        config: Temporal split proportions and unknown-entity policy.

    Returns:
        Temporal splits and filtering statistics.

    Raises:
        ValueError: If required columns are missing, the input is empty or
            there are not enough rows to populate all splits.
    """
    _validate_interactions(interactions)

    sorted_interactions = interactions.sort_values(
        ["last_interaction_at", "user_id", "item_id"],
    ).reset_index(drop=True)

    train_end, validation_end = _calculate_boundaries(
        row_count=len(sorted_interactions),
        config=config,
    )

    train = sorted_interactions.iloc[:train_end].copy()
    validation = sorted_interactions.iloc[train_end:validation_end].copy()
    test = sorted_interactions.iloc[validation_end:].copy()

    validation_removed = 0
    test_removed = 0

    if config.filter_unknown_entities:
        validation, validation_removed = _filter_known_entities(
            candidate=validation,
            train=train,
        )
        test, test_removed = _filter_known_entities(
            candidate=test,
            train=train,
        )

    return TemporalSplitResult(
        train=train.reset_index(drop=True),
        validation=validation.reset_index(drop=True),
        test=test.reset_index(drop=True),
        validation_removed_unknowns=validation_removed,
        test_removed_unknowns=test_removed,
    )


def _calculate_boundaries(
    row_count: int,
    config: TemporalSplitConfig,
) -> tuple[int, int]:
    """Calculate chronological split boundaries."""
    train_end = int(row_count * config.train_size)
    validation_count = int(row_count * config.validation_size)
    validation_end = train_end + validation_count

    if train_end == 0:
        msg = "Temporal split produced an empty training set."
        raise ValueError(msg)

    if validation_count == 0:
        msg = "Temporal split produced an empty validation set."
        raise ValueError(msg)

    if validation_end >= row_count:
        msg = "Temporal split produced an empty test set."
        raise ValueError(msg)

    return train_end, validation_end


def _filter_known_entities(
    candidate: pd.DataFrame,
    train: pd.DataFrame,
) -> tuple[pd.DataFrame, int]:
    """Keep only users and items observed in training."""
    known_users = set(train["user_id"].unique())
    known_items = set(train["item_id"].unique())

    known_mask = candidate["user_id"].isin(known_users) & candidate["item_id"].isin(
        known_items
    )

    filtered = candidate.loc[known_mask].copy()
    removed_count = len(candidate) - len(filtered)

    return filtered, removed_count


def _validate_interactions(interactions: pd.DataFrame) -> None:
    """Validate interactions before chronological splitting."""
    if interactions.empty:
        msg = "Cannot split an empty interaction dataset."
        raise ValueError(msg)

    missing_columns = REQUIRED_INTERACTION_COLUMNS - set(interactions.columns)

    if missing_columns:
        missing = ", ".join(sorted(missing_columns))
        msg = f"Missing required interaction columns: {missing}."
        raise ValueError(msg)

    if interactions["last_interaction_at"].isna().any():
        msg = "last_interaction_at cannot contain missing values."
        raise ValueError(msg)

    if not is_datetime64_any_dtype(
        interactions["last_interaction_at"],
    ):
        msg = "last_interaction_at must be a datetime column."
        raise ValueError(msg)
