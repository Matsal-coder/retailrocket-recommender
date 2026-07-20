"""Tests for user-item interaction construction."""

import pandas as pd
import pytest

from retail_recommender.features.interaction_builder import (
    InteractionFilterConfig,
    build_and_filter_interactions,
    build_interactions,
    filter_interactions,
)

VIEW_WEIGHT = 1.0
CART_WEIGHT = 3.0
TRANSACTION_WEIGHT = 5.0

USER_A = 10
USER_B = 20

ITEM_X = 100
ITEM_Y = 200
ITEM_Z = 300

EXPECTED_PAIR_COUNT = 3
EXPECTED_AGGREGATED_SCORE = VIEW_WEIGHT + CART_WEIGHT
EXPECTED_AGGREGATED_COUNT = 2


def make_clean_events() -> pd.DataFrame:
    """Create clean event data for interaction tests."""
    return pd.DataFrame(
        {
            "user_id": [USER_A, USER_A, USER_A, USER_B],
            "item_id": [ITEM_X, ITEM_X, ITEM_Y, ITEM_Z],
            "event_type": [
                "view",
                "addtocart",
                "view",
                "transaction",
            ],
            "event_weight": [
                VIEW_WEIGHT,
                CART_WEIGHT,
                VIEW_WEIGHT,
                TRANSACTION_WEIGHT,
            ],
            "timestamp": [
                1_600_000_000_000,
                1_600_000_001_000,
                1_600_000_002_000,
                1_600_000_003_000,
            ],
            "datetime": pd.to_datetime(
                [
                    1_600_000_000_000,
                    1_600_000_001_000,
                    1_600_000_002_000,
                    1_600_000_003_000,
                ],
                unit="ms",
                utc=True,
            ),
        }
    )


def test_build_interactions_aggregates_user_item_events() -> None:
    """It should aggregate repeated events for each user-item pair."""
    events = make_clean_events()

    result = build_interactions(events)

    assert len(result) == EXPECTED_PAIR_COUNT

    aggregated_pair = result[
        (result["user_id"] == USER_A) & (result["item_id"] == ITEM_X)
    ].iloc[0]

    assert aggregated_pair["interaction_score"] == EXPECTED_AGGREGATED_SCORE
    assert aggregated_pair["interaction_count"] == EXPECTED_AGGREGATED_COUNT
    assert aggregated_pair["target"] == 1


def test_build_interactions_uses_latest_interaction_datetime() -> None:
    """It should preserve the latest event time for each pair."""
    events = make_clean_events()

    result = build_interactions(events)

    aggregated_pair = result[
        (result["user_id"] == USER_A) & (result["item_id"] == ITEM_X)
    ].iloc[0]

    expected_datetime = pd.Timestamp(
        1_600_000_001_000,
        unit="ms",
        tz="UTC",
    )

    assert aggregated_pair["last_interaction_at"] == expected_datetime


def test_build_interactions_sets_positive_target() -> None:
    """It should mark every observed pair as a positive interaction."""
    result = build_interactions(make_clean_events())

    assert result["target"].eq(1).all()


def test_build_interactions_raises_error_for_empty_events() -> None:
    """It should reject an empty event dataset."""
    with pytest.raises(ValueError, match="empty event dataset"):
        build_interactions(pd.DataFrame())


def test_build_interactions_raises_error_for_missing_columns() -> None:
    """It should identify missing clean-event columns."""
    invalid_events = pd.DataFrame(
        {
            "user_id": [USER_A],
            "item_id": [ITEM_X],
        }
    )

    with pytest.raises(ValueError, match="datetime"):
        build_interactions(invalid_events)


def test_filter_interactions_removes_inactive_users_and_items() -> None:
    """It should retain only users and items above both thresholds."""
    interactions = pd.DataFrame(
        {
            "user_id": [
                USER_A,
                USER_A,
                USER_B,
                USER_B,
            ],
            "item_id": [
                ITEM_X,
                ITEM_Y,
                ITEM_X,
                ITEM_Z,
            ],
            "interaction_score": [
                VIEW_WEIGHT,
                VIEW_WEIGHT,
                VIEW_WEIGHT,
                VIEW_WEIGHT,
            ],
            "interaction_count": [1, 1, 1, 1],
            "last_interaction_at": pd.to_datetime(
                [
                    "2026-01-01",
                    "2026-01-02",
                    "2026-01-03",
                    "2026-01-04",
                ],
                utc=True,
            ),
            "target": [1, 1, 1, 1],
        }
    )
    config = InteractionFilterConfig(
        minimum_user_interactions=2,
        minimum_item_interactions=2,
    )

    result = filter_interactions(interactions, config)

    assert result.empty


def test_filter_interactions_preserves_valid_core() -> None:
    """It should preserve interactions that meet both thresholds."""
    interactions = pd.DataFrame(
        {
            "user_id": [
                USER_A,
                USER_A,
                USER_B,
                USER_B,
            ],
            "item_id": [
                ITEM_X,
                ITEM_Y,
                ITEM_X,
                ITEM_Y,
            ],
            "interaction_score": [
                VIEW_WEIGHT,
                VIEW_WEIGHT,
                VIEW_WEIGHT,
                VIEW_WEIGHT,
            ],
            "interaction_count": [1, 1, 1, 1],
            "last_interaction_at": pd.to_datetime(
                [
                    "2026-01-01",
                    "2026-01-02",
                    "2026-01-03",
                    "2026-01-04",
                ],
                utc=True,
            ),
            "target": [1, 1, 1, 1],
        }
    )
    config = InteractionFilterConfig(
        minimum_user_interactions=2,
        minimum_item_interactions=2,
    )

    result = filter_interactions(interactions, config)

    assert len(result) == len(interactions)


def test_build_and_filter_interactions_combines_both_steps() -> None:
    """It should aggregate events before applying activity filters."""
    config = InteractionFilterConfig(
        minimum_user_interactions=1,
        minimum_item_interactions=1,
    )

    result = build_and_filter_interactions(
        events=make_clean_events(),
        config=config,
    )

    assert len(result) == EXPECTED_PAIR_COUNT


@pytest.mark.parametrize(
    ("field_name", "field_value"),
    [
        ("minimum_user_interactions", 0),
        ("minimum_item_interactions", 0),
    ],
)
def test_filter_config_rejects_invalid_minimums(
    field_name: str,
    field_value: int,
) -> None:
    """It should reject minimum values below one."""
    config_values = {
        "minimum_user_interactions": 1,
        "minimum_item_interactions": 1,
    }
    config_values[field_name] = field_value

    with pytest.raises(ValueError, match="at least 1"):
        InteractionFilterConfig(**config_values)
