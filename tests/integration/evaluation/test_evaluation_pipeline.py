"""Integration test for baseline recommendation evaluation."""

import json
from pathlib import Path

import pandas as pd

from retail_recommender.evaluation.evaluator import (
    RecommenderEvaluator,
)
from retail_recommender.evaluation.reports import (
    write_evaluation_report,
)
from retail_recommender.models.popularity import (
    PopularityRecommender,
)

EVALUATION_K = 2
EVALUATED_USERS = 2


def test_popularity_model_generates_evaluation_report(
    tmp_path: Path,
) -> None:
    train_interactions = pd.DataFrame(
        {
            "user_idx": [0, 0, 1, 1, 2],
            "item_idx": [10, 20, 10, 30, 20],
            "interaction_score": [5.0, 1.0, 3.0, 4.0, 2.0],
            "interaction_count": [1, 1, 1, 1, 1],
        }
    )

    model = PopularityRecommender().fit(train_interactions)

    evaluator = RecommenderEvaluator(
        model,
        k=EVALUATION_K,
        catalog_items={10, 20, 30, 40},
        exclude_seen_items=True,
    )

    result = evaluator.evaluate(
        relevant_items_by_user={
            0: {30},
            1: {20},
        },
        seen_items_by_user={
            0: {10, 20},
            1: {10, 30},
        },
    )

    report_path = tmp_path / "test_metrics.json"

    write_evaluation_report(
        result,
        report_path,
        model_name="popularity",
        metadata={
            "split": "test",
        },
    )

    payload = json.loads(report_path.read_text(encoding="utf-8"))

    assert report_path.exists()
    assert result.evaluated_users == EVALUATED_USERS
    assert payload["model_name"] == "popularity"
    assert payload["metrics"]["k"] == EVALUATION_K
