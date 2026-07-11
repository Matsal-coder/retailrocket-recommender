from pathlib import Path

import pandas as pd
import pytest

from retail_recommender.data.loaders.retailrocket_loader import load_events


def test_load_events_returns_dataframe(tmp_path: Path) -> None:
    events_path = tmp_path / "events.csv"
    expected_events = pd.DataFrame(
        {
            "timestamp": [1433221332117, 1433224214164],
            "visitorid": [257597, 992329],
            "event": ["view", "addtocart"],
            "itemid": [355908, 248676],
            "transactionid": [None, None],
        }
    )
    expected_events.to_csv(events_path, index=False)

    events = load_events(events_path)

    assert isinstance(events, pd.DataFrame)
    assert events.shape == (2, 5)
    assert list(events.columns) == [
        "timestamp",
        "visitorid",
        "event",
        "itemid",
        "transactionid",
    ]


def test_load_events_raises_file_not_found_error_for_missing_file(
    tmp_path: Path,
) -> None:
    missing_path = tmp_path / "missing_events.csv"

    with pytest.raises(FileNotFoundError, match="Events file not found"):
        load_events(missing_path)


def test_load_events_raises_value_error_for_empty_file(tmp_path: Path) -> None:
    events_path = tmp_path / "events.csv"
    events_path.write_text("", encoding="utf-8")

    with pytest.raises(ValueError, match="Events file is empty"):
        load_events(events_path)
