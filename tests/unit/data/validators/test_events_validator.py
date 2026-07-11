import pandas as pd

from retail_recommender.data.validators.events_validator import (
    EventsValidationConfig,
    validate_events,
)

EXPECTED_ROWS = 3
EXPECTED_UNIQUE_USERS = 2
EXPECTED_UNIQUE_ITEMS = 3
MINIMUM_INTERACTIONS_FOR_VALID_FIXTURE = 3
MINIMUM_USERS_FOR_VALID_FIXTURE = 2
MINIMUM_ITEMS_FOR_VALID_FIXTURE = 3
MINIMUM_INTERACTIONS_TOO_HIGH = 10
MINIMUM_USERS_TOO_HIGH = 10
MINIMUM_ITEMS_TOO_HIGH = 10


def test_validate_events_returns_valid_result_for_valid_dataset() -> None:
    events = _make_valid_events()

    result = validate_events(
        events,
        config=EventsValidationConfig(
            minimum_interactions=MINIMUM_INTERACTIONS_FOR_VALID_FIXTURE,
            minimum_users=MINIMUM_USERS_FOR_VALID_FIXTURE,
            minimum_items=MINIMUM_ITEMS_FOR_VALID_FIXTURE,
        ),
    )

    assert result.is_valid is True
    assert result.errors == []
    assert result.summary["rows"] == EXPECTED_ROWS
    assert result.summary["unique_users"] == EXPECTED_UNIQUE_USERS
    assert result.summary["unique_items"] == EXPECTED_UNIQUE_ITEMS
    assert result.summary["event_counts"] == {
        "view": 1,
        "addtocart": 1,
        "transaction": 1,
    }


def test_validate_events_fails_when_required_column_is_missing() -> None:
    events = _make_valid_events().drop(columns=["transactionid"])

    result = validate_events(
        events,
        config=EventsValidationConfig(
            minimum_interactions=MINIMUM_INTERACTIONS_FOR_VALID_FIXTURE,
            minimum_users=MINIMUM_USERS_FOR_VALID_FIXTURE,
            minimum_items=MINIMUM_ITEMS_FOR_VALID_FIXTURE,
        ),
    )

    assert result.is_valid is False
    assert result.errors == ["Missing required columns: transactionid"]
    assert result.summary["rows"] == EXPECTED_ROWS


def test_validate_events_fails_when_required_column_is_fully_empty() -> None:
    events = _make_valid_events()
    events["visitorid"] = None

    result = validate_events(
        events,
        config=EventsValidationConfig(
            minimum_interactions=MINIMUM_INTERACTIONS_FOR_VALID_FIXTURE,
            minimum_users=MINIMUM_USERS_FOR_VALID_FIXTURE,
            minimum_items=MINIMUM_ITEMS_FOR_VALID_FIXTURE,
        ),
    )

    assert result.is_valid is False
    assert "Required columns are fully empty: visitorid" in result.errors


def test_validate_events_fails_for_unexpected_event_values() -> None:
    events = _make_valid_events()
    events.loc[0, "event"] = "purchase"

    result = validate_events(
        events,
        config=EventsValidationConfig(
            minimum_interactions=MINIMUM_INTERACTIONS_FOR_VALID_FIXTURE,
            minimum_users=MINIMUM_USERS_FOR_VALID_FIXTURE,
            minimum_items=MINIMUM_ITEMS_FOR_VALID_FIXTURE,
        ),
    )

    assert result.is_valid is False
    assert "Unexpected event values found: purchase" in result.errors


def test_validate_events_fails_when_timestamp_cannot_be_converted() -> None:
    events = _make_valid_events()
    events["timestamp"] = ["invalid", "invalid", "invalid"]

    result = validate_events(
        events,
        config=EventsValidationConfig(
            minimum_interactions=MINIMUM_INTERACTIONS_FOR_VALID_FIXTURE,
            minimum_users=MINIMUM_USERS_FOR_VALID_FIXTURE,
            minimum_items=MINIMUM_ITEMS_FOR_VALID_FIXTURE,
        ),
    )

    assert result.is_valid is False
    assert "Timestamp column cannot be converted to datetime." in result.errors


def test_validate_events_fails_when_dataset_has_few_interactions() -> None:
    events = _make_valid_events()

    result = validate_events(
        events,
        config=EventsValidationConfig(
            minimum_interactions=MINIMUM_INTERACTIONS_TOO_HIGH,
            minimum_users=MINIMUM_USERS_FOR_VALID_FIXTURE,
            minimum_items=MINIMUM_ITEMS_FOR_VALID_FIXTURE,
        ),
    )

    assert result.is_valid is False
    assert "Dataset has 3 interactions, but at least 10 are required." in result.errors


def test_validate_events_fails_when_dataset_has_few_users() -> None:
    events = _make_valid_events()

    result = validate_events(
        events,
        config=EventsValidationConfig(
            minimum_interactions=MINIMUM_INTERACTIONS_FOR_VALID_FIXTURE,
            minimum_users=MINIMUM_USERS_TOO_HIGH,
            minimum_items=MINIMUM_ITEMS_FOR_VALID_FIXTURE,
        ),
    )

    assert result.is_valid is False
    assert "Dataset has 2 unique users, but at least 10 are required." in result.errors


def test_validate_events_fails_when_dataset_has_few_items() -> None:
    events = _make_valid_events()

    result = validate_events(
        events,
        config=EventsValidationConfig(
            minimum_interactions=MINIMUM_INTERACTIONS_FOR_VALID_FIXTURE,
            minimum_users=MINIMUM_USERS_FOR_VALID_FIXTURE,
            minimum_items=MINIMUM_ITEMS_TOO_HIGH,
        ),
    )

    assert result.is_valid is False
    assert "Dataset has 3 unique items, but at least 10 are required." in result.errors


def test_validation_result_can_be_converted_to_dict() -> None:
    events = _make_valid_events()

    result = validate_events(
        events,
        config=EventsValidationConfig(
            minimum_interactions=MINIMUM_INTERACTIONS_FOR_VALID_FIXTURE,
            minimum_users=MINIMUM_USERS_FOR_VALID_FIXTURE,
            minimum_items=MINIMUM_ITEMS_FOR_VALID_FIXTURE,
        ),
    )

    result_as_dict = result.to_dict()

    assert result_as_dict["is_valid"] is True
    assert result_as_dict["errors"] == []
    assert result_as_dict["warnings"] == []
    assert result_as_dict["summary"]["rows"] == EXPECTED_ROWS


def _make_valid_events() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "timestamp": [
                1433221332117,
                1433224214164,
                1433221999827,
            ],
            "visitorid": [
                257597,
                992329,
                257597,
            ],
            "event": [
                "view",
                "addtocart",
                "transaction",
            ],
            "itemid": [
                355908,
                248676,
                355909,
            ],
            "transactionid": [
                None,
                None,
                4000,
            ],
        }
    )
