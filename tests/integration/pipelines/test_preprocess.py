"""Integration tests for the event preprocessing pipeline."""

from pathlib import Path

import pandas as pd
import yaml

from retail_recommender.pipelines.preprocess import (
    run_preprocessing_pipeline,
)

EXPECTED_CLEAN_EVENT_COUNT = 3

EVENT_WEIGHTS = {
    "view": 1.0,
    "addtocart": 3.0,
    "transaction": 5.0,
}


def test_preprocessing_pipeline_generates_clean_parquet(
    tmp_path: Path,
) -> None:
    """It should load, preprocess and persist clean event data."""
    raw_path = tmp_path / "data" / "raw" / "events.csv"
    output_path = tmp_path / "data" / "interim" / "events_clean.parquet"
    config_path = tmp_path / "configs" / "data.yaml"
    params_path = tmp_path / "params.yaml"

    raw_path.parent.mkdir(parents=True)
    config_path.parent.mkdir(parents=True)

    raw_events = pd.DataFrame(
        {
            "timestamp": [
                1_600_000_000_000,
                1_600_000_001_000,
                1_600_000_002_000,
                1_600_000_003_000,
                1_600_000_004_000,
            ],
            "visitorid": [10, 10, 20, None, 30],
            "event": [
                "view",
                "addtocart",
                "transaction",
                "view",
                "wishlist",
            ],
            "itemid": [100, 100, 200, 300, 400],
            "transactionid": [None, None, 999, None, None],
        },
    )
    raw_events.to_csv(raw_path, index=False)

    data_config = {
        "raw_events_path": str(raw_path),
        "interim_events_path": str(output_path),
        "validation_report_path": str(
            tmp_path / "artifacts" / "reports" / "data_validation.json"
        ),
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
        "minimum_interactions": 1,
        "minimum_users": 1,
        "minimum_items": 1,
    }

    pipeline_params = {
        "data": {
            "minimum_interactions": 1,
            "minimum_users": 1,
            "minimum_items": 1,
        },
        "preprocessing": {
            "strategy": "implicit_feedback",
            "allowed_event_types": list(EVENT_WEIGHTS),
            "event_weights": EVENT_WEIGHTS,
        },
    }

    config_path.write_text(
        yaml.safe_dump(data_config),
        encoding="utf-8",
    )
    params_path.write_text(
        yaml.safe_dump(pipeline_params),
        encoding="utf-8",
    )

    result = run_preprocessing_pipeline(
        data_config_path=config_path,
        params_path=params_path,
    )

    assert output_path.exists()
    assert len(result) == EXPECTED_CLEAN_EVENT_COUNT

    saved_events = pd.read_parquet(output_path)

    assert len(saved_events) == EXPECTED_CLEAN_EVENT_COUNT
    assert saved_events.columns.tolist() == [
        "user_id",
        "item_id",
        "event_type",
        "event_weight",
        "timestamp",
        "datetime",
    ]
    assert saved_events["event_type"].tolist() == [
        "view",
        "addtocart",
        "transaction",
    ]
    assert saved_events["event_weight"].tolist() == [
        EVENT_WEIGHTS["view"],
        EVENT_WEIGHTS["addtocart"],
        EVENT_WEIGHTS["transaction"],
    ]
