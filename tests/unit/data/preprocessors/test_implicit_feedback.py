"""Tests for the implicit-feedback preprocessing strategy."""

import pandas as pd
import pytest
from pandas.api.types import is_datetime64_any_dtype

from retail_recommender.data.preprocessors.implicit_feedback import (
    ImplicitFeedbackPreprocessor,
)

EVENT_WEIGHTS = {
    "view": 1.0,
    "addtocart": 3.0,
    "transaction": 5.0,
}

EXPECTED_OUTPUT_COLUMNS = [
    "user_id",
    "item_id",
    "event_type",
    "event_weight",
    "timestamp",
    "datetime",
]

EXPECTED_USER_ID = 10


@pytest.fixture
def preprocessor() -> ImplicitFeedbackPreprocessor:
    """Create a configured implicit-feedback preprocessor."""
    return ImplicitFeedbackPreprocessor(
        event_weights=EVENT_WEIGHTS,
        allowed_event_types=list(EVENT_WEIGHTS),
    )


def test_transform_standardizes_raw_retailrocket_events(
    preprocessor: ImplicitFeedbackPreprocessor,
) -> None:
    """It should standardize raw columns and map event weights."""
    raw_events = pd.DataFrame(
        {
            "timestamp": [1_600_000_000_000, 1_600_000_001_000],
            "visitorid": [10, 20],
            "event": ["view", "transaction"],
            "itemid": [100, 200],
            "transactionid": [None, 999],
        },
    )

    result = preprocessor.transform(raw_events)

    assert result.columns.tolist() == EXPECTED_OUTPUT_COLUMNS
    assert result["user_id"].tolist() == [10, 20]
    assert result["item_id"].tolist() == [100, 200]
    assert result["event_type"].tolist() == ["view", "transaction"]
    assert result["event_weight"].tolist() == [1.0, 5.0]
    assert is_datetime64_any_dtype(result["datetime"])
    assert str(result["datetime"].dt.tz) == "UTC"


def test_transform_removes_invalid_and_unsupported_events(
    preprocessor: ImplicitFeedbackPreprocessor,
) -> None:
    """It should remove missing IDs and unsupported event types."""
    raw_events = pd.DataFrame(
        {
            "timestamp": [
                1_600_000_000_000,
                1_600_000_001_000,
                1_600_000_002_000,
            ],
            "visitorid": [10, None, 30],
            "event": [" VIEW ", "addtocart", "wishlist"],
            "itemid": [100, 200, 300],
        },
    )

    result = preprocessor.transform(raw_events)

    assert len(result) == 1
    assert result.loc[0, "user_id"] == EXPECTED_USER_ID
    assert result.loc[0, "event_type"] == "view"
    assert result.loc[0, "event_weight"] == 1.0


def test_transform_orders_events_by_timestamp(
    preprocessor: ImplicitFeedbackPreprocessor,
) -> None:
    """It should return events in chronological order."""
    raw_events = pd.DataFrame(
        {
            "timestamp": [1_600_000_002_000, 1_600_000_000_000],
            "visitorid": [20, 10],
            "event": ["view", "view"],
            "itemid": [200, 100],
        },
    )

    result = preprocessor.transform(raw_events)

    assert result["timestamp"].tolist() == [
        1_600_000_000_000,
        1_600_000_002_000,
    ]


def test_transform_accepts_already_standardized_columns(
    preprocessor: ImplicitFeedbackPreprocessor,
) -> None:
    """It should accept events using the project's internal column names."""
    standardized_events = pd.DataFrame(
        {
            "timestamp": [1_600_000_000_000],
            "user_id": [10],
            "event_type": ["addtocart"],
            "item_id": [100],
        },
    )

    result = preprocessor.transform(standardized_events)

    assert result.loc[0, "event_weight"] == EVENT_WEIGHTS["addtocart"]


def test_transform_raises_error_when_required_columns_are_missing(
    preprocessor: ImplicitFeedbackPreprocessor,
) -> None:
    """It should fail clearly when required columns are absent."""
    invalid_events = pd.DataFrame(
        {
            "timestamp": [1_600_000_000_000],
            "visitorid": [10],
            "event": ["view"],
        },
    )

    with pytest.raises(ValueError, match="item_id"):
        preprocessor.transform(invalid_events)


def test_initialization_raises_error_when_event_weights_are_empty() -> None:
    """It should require at least one event weight."""
    with pytest.raises(ValueError, match="At least one event weight"):
        ImplicitFeedbackPreprocessor(event_weights={})


def test_initialization_raises_error_for_allowed_event_without_weight() -> None:
    """It should reject allowed events without configured weights."""
    with pytest.raises(ValueError, match="transaction"):
        ImplicitFeedbackPreprocessor(
            event_weights={"view": 1.0},
            allowed_event_types=["view", "transaction"],
        )
