import json
from pathlib import Path

import pandas as pd
import pytest
import yaml

from retail_recommender.pipelines.validate_data import run_validate_data_pipeline

EXPECTED_ROWS = 3
EXPECTED_UNIQUE_USERS = 2
EXPECTED_UNIQUE_ITEMS = 3


def test_validate_data_pipeline_creates_validation_report(
    tmp_path: Path,
) -> None:
    raw_events_path = tmp_path / "data" / "raw" / "events.csv"
    validation_report_path = tmp_path / "artifacts" / "reports" / "data_validation.json"
    config_path = tmp_path / "configs" / "data.yaml"

    raw_events_path.parent.mkdir(parents=True)
    config_path.parent.mkdir(parents=True)

    _make_valid_events().to_csv(raw_events_path, index=False)

    config = {
        "raw_events_path": str(raw_events_path),
        "validation_report_path": str(validation_report_path),
        "required_columns": [
            "timestamp",
            "visitorid",
            "event",
            "itemid",
            "transactionid",
        ],
        "allowed_events": [
            "view",
            "addtocart",
            "transaction",
        ],
        "minimum_interactions": EXPECTED_ROWS,
        "minimum_users": EXPECTED_UNIQUE_USERS,
        "minimum_items": EXPECTED_UNIQUE_ITEMS,
    }

    with config_path.open("w", encoding="utf-8") as config_file:
        yaml.safe_dump(config, config_file)

    result = run_validate_data_pipeline(config_path=config_path)

    assert result.is_valid is True
    assert validation_report_path.exists()

    report = json.loads(validation_report_path.read_text(encoding="utf-8"))

    assert report["is_valid"] is True
    assert report["errors"] == []
    assert report["summary"]["rows"] == EXPECTED_ROWS
    assert report["summary"]["unique_users"] == EXPECTED_UNIQUE_USERS
    assert report["summary"]["unique_items"] == EXPECTED_UNIQUE_ITEMS


def test_validate_data_pipeline_raises_error_for_invalid_dataset(
    tmp_path: Path,
) -> None:
    raw_events_path = tmp_path / "data" / "raw" / "events.csv"
    validation_report_path = tmp_path / "artifacts" / "reports" / "data_validation.json"
    config_path = tmp_path / "configs" / "data.yaml"

    raw_events_path.parent.mkdir(parents=True)
    config_path.parent.mkdir(parents=True)

    invalid_events = _make_valid_events()
    invalid_events["event"] = "invalid_event"
    invalid_events.to_csv(raw_events_path, index=False)

    config = {
        "raw_events_path": str(raw_events_path),
        "validation_report_path": str(validation_report_path),
        "required_columns": [
            "timestamp",
            "visitorid",
            "event",
            "itemid",
            "transactionid",
        ],
        "allowed_events": [
            "view",
            "addtocart",
            "transaction",
        ],
        "minimum_interactions": EXPECTED_ROWS,
        "minimum_users": EXPECTED_UNIQUE_USERS,
        "minimum_items": EXPECTED_UNIQUE_ITEMS,
    }

    with config_path.open("w", encoding="utf-8") as config_file:
        yaml.safe_dump(config, config_file)

    with pytest.raises(ValueError, match="Data validation failed"):
        run_validate_data_pipeline(config_path=config_path)

    assert validation_report_path.exists()

    report = json.loads(validation_report_path.read_text(encoding="utf-8"))

    assert report["is_valid"] is False
    assert "Unexpected event values found: invalid_event" in report["errors"]


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
