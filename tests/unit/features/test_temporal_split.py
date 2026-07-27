"""Tests for chronological interaction splitting."""

import pandas as pd
import pytest

from retail_recommender.features.temporal_split import (
    TemporalSplitConfig,
    temporal_split,
)

TOTAL_INTERACTIONS = 20
EXPECTED_TRAIN_SIZE = 14
EXPECTED_VALIDATION_SIZE = 3
EXPECTED_TEST_SIZE = 3

USER_A = 10
USER_B = 20
USER_UNKNOWN = 99

ITEM_X = 100
ITEM_Y = 200
ITEM_UNKNOWN = 999


def make_interactions(row_count: int = TOTAL_INTERACTIONS) -> pd.DataFrame:
    """Create chronologically ordered interactions for split tests."""
    return pd.DataFrame(
        {
            "user_id": [
                USER_A if index % 2 == 0 else USER_B for index in range(row_count)
            ],
            "item_id": [
                ITEM_X if index % 2 == 0 else ITEM_Y for index in range(row_count)
            ],
            "interaction_score": [1.0] * row_count,
            "interaction_count": [1] * row_count,
            "last_interaction_at": pd.date_range(
                start="2026-01-01",
                periods=row_count,
                freq="D",
                tz="UTC",
            ),
            "target": [1] * row_count,
        }
    )


def make_split_config(
    filter_unknown_entities: bool = True,
) -> TemporalSplitConfig:
    """Create the default temporal split configuration."""
    return TemporalSplitConfig(
        train_size=0.70,
        validation_size=0.15,
        test_size=0.15,
        filter_unknown_entities=filter_unknown_entities,
    )


def test_temporal_split_creates_expected_split_sizes() -> None:
    """It should split rows according to chronological proportions."""
    result = temporal_split(
        interactions=make_interactions(),
        config=make_split_config(),
    )

    assert len(result.train) == EXPECTED_TRAIN_SIZE
    assert len(result.validation) == EXPECTED_VALIDATION_SIZE
    assert len(result.test) == EXPECTED_TEST_SIZE


def test_temporal_split_preserves_chronological_order() -> None:
    """Train must precede validation, which must precede test."""
    result = temporal_split(
        interactions=make_interactions(),
        config=make_split_config(),
    )

    train_max = result.train["last_interaction_at"].max()
    validation_min = result.validation["last_interaction_at"].min()
    validation_max = result.validation["last_interaction_at"].max()
    test_min = result.test["last_interaction_at"].min()

    assert train_max < validation_min
    assert validation_max < test_min


def test_temporal_split_sorts_unsorted_input() -> None:
    """It should sort the input before splitting."""
    interactions = make_interactions().sample(
        frac=1.0,
        random_state=17,
    )

    result = temporal_split(
        interactions=interactions,
        config=make_split_config(),
    )

    assert result.train["last_interaction_at"].is_monotonic_increasing
    assert result.validation["last_interaction_at"].is_monotonic_increasing
    assert result.test["last_interaction_at"].is_monotonic_increasing


def test_temporal_split_filters_unknown_validation_entities() -> None:
    """It should remove validation pairs unknown to training."""
    interactions = make_interactions()
    interactions.loc[
        EXPECTED_TRAIN_SIZE,
        ["user_id", "item_id"],
    ] = [USER_UNKNOWN, ITEM_UNKNOWN]

    result = temporal_split(
        interactions=interactions,
        config=make_split_config(),
    )

    assert result.validation_removed_unknowns == 1
    assert USER_UNKNOWN not in result.validation["user_id"].values
    assert ITEM_UNKNOWN not in result.validation["item_id"].values


def test_temporal_split_filters_unknown_test_entities() -> None:
    """It should remove test pairs unknown to training."""
    interactions = make_interactions()
    interactions.loc[
        TOTAL_INTERACTIONS - 1,
        ["user_id", "item_id"],
    ] = [USER_UNKNOWN, ITEM_UNKNOWN]

    result = temporal_split(
        interactions=interactions,
        config=make_split_config(),
    )

    assert result.test_removed_unknowns == 1
    assert USER_UNKNOWN not in result.test["user_id"].values
    assert ITEM_UNKNOWN not in result.test["item_id"].values


def test_temporal_split_can_preserve_unknown_entities() -> None:
    """It should keep unknowns when filtering is disabled."""
    interactions = make_interactions()
    interactions.loc[
        TOTAL_INTERACTIONS - 1,
        ["user_id", "item_id"],
    ] = [USER_UNKNOWN, ITEM_UNKNOWN]

    result = temporal_split(
        interactions=interactions,
        config=make_split_config(filter_unknown_entities=False),
    )

    assert result.test_removed_unknowns == 0
    assert USER_UNKNOWN in result.test["user_id"].values
    assert ITEM_UNKNOWN in result.test["item_id"].values


def test_temporal_split_raises_error_for_empty_dataset() -> None:
    """It should reject an empty interaction dataset."""
    with pytest.raises(ValueError, match="empty interaction dataset"):
        temporal_split(
            interactions=pd.DataFrame(),
            config=make_split_config(),
        )


def test_temporal_split_raises_error_for_missing_columns() -> None:
    """It should reject data without the temporal column."""
    interactions = pd.DataFrame(
        {
            "user_id": [USER_A],
            "item_id": [ITEM_X],
        }
    )

    with pytest.raises(ValueError, match="last_interaction_at"):
        temporal_split(
            interactions=interactions,
            config=make_split_config(),
        )


def test_temporal_split_rejects_missing_datetime_values() -> None:
    """It should reject missing temporal values."""
    interactions = make_interactions()
    interactions.loc[0, "last_interaction_at"] = pd.NaT

    with pytest.raises(ValueError, match="cannot contain missing"):
        temporal_split(
            interactions=interactions,
            config=make_split_config(),
        )


def test_temporal_split_rejects_non_datetime_column() -> None:
    """It should require a datetime temporal column."""
    interactions = make_interactions()
    interactions["last_interaction_at"] = interactions["last_interaction_at"].astype(
        "string"
    )

    with pytest.raises(ValueError, match="must be a datetime"):
        temporal_split(
            interactions=interactions,
            config=make_split_config(),
        )


@pytest.mark.parametrize(
    ("train_size", "validation_size", "test_size"),
    [
        (0.70, 0.15, 0.10),
        (0.70, 0.00, 0.30),
        (1.00, 0.00, 0.00),
        (-0.10, 0.50, 0.60),
    ],
)
def test_split_config_rejects_invalid_proportions(
    train_size: float,
    validation_size: float,
    test_size: float,
) -> None:
    """It should reject invalid split proportions."""
    with pytest.raises(ValueError):
        TemporalSplitConfig(
            train_size=train_size,
            validation_size=validation_size,
            test_size=test_size,
        )


def test_temporal_split_rejects_dataset_too_small() -> None:
    """It should reject datasets that cannot populate every split."""
    interactions = make_interactions(row_count=2)

    with pytest.raises(ValueError, match="empty validation set"):
        temporal_split(
            interactions=interactions,
            config=make_split_config(),
        )
