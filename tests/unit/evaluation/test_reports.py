"""Unit tests for evaluation report writers."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest

from retail_recommender.evaluation.evaluator import (
    EvaluationResult,
)
from retail_recommender.evaluation.reports import (
    write_evaluation_report,
    write_model_comparison,
)

DEFAULT_K = 10
DEFAULT_USER_COUNT = 3
LEN_ROWS = 2


@pytest.fixture
def evaluation_result() -> EvaluationResult:
    """Create a deterministic evaluation result."""
    return EvaluationResult(
        k=DEFAULT_K,
        evaluated_users=DEFAULT_USER_COUNT,
        precision_at_k=0.2,
        recall_at_k=0.4,
        ndcg_at_k=0.35,
        map_at_k=0.3,
        coverage_at_k=0.5,
    )


def test_write_evaluation_report_creates_json(
    evaluation_result: EvaluationResult,
    tmp_path: Path,
) -> None:
    report_path = tmp_path / "reports" / "metrics.json"

    write_evaluation_report(
        evaluation_result,
        report_path,
        model_name="popularity",
        metadata={
            "split": "test",
        },
    )

    payload = json.loads(report_path.read_text(encoding="utf-8"))

    assert payload["model_name"] == "popularity"
    assert payload["metrics"]["k"] == DEFAULT_K
    assert payload["metrics"]["ndcg_at_k"] == pytest.approx(0.35)
    assert payload["metadata"]["split"] == "test"


def test_write_evaluation_report_rejects_empty_model_name(
    evaluation_result: EvaluationResult,
    tmp_path: Path,
) -> None:
    with pytest.raises(
        ValueError,
        match="model_name must not be empty",
    ):
        write_evaluation_report(
            evaluation_result,
            tmp_path / "metrics.json",
            model_name=" ",
        )


def test_write_model_comparison_creates_ordered_csv(
    evaluation_result: EvaluationResult,
    tmp_path: Path,
) -> None:
    better_result = EvaluationResult(
        k=DEFAULT_K,
        evaluated_users=DEFAULT_USER_COUNT,
        precision_at_k=0.3,
        recall_at_k=0.5,
        ndcg_at_k=0.6,
        map_at_k=0.4,
        coverage_at_k=0.55,
    )
    comparison_path = tmp_path / "reports" / "comparison.csv"

    write_model_comparison(
        {
            "popularity": evaluation_result,
            "item_knn": better_result,
        },
        comparison_path,
    )

    with comparison_path.open(
        encoding="utf-8",
        newline="",
    ) as comparison_file:
        rows = list(csv.DictReader(comparison_file))

    assert len(rows) == LEN_ROWS
    assert rows[0]["model_name"] == "item_knn"
    assert float(rows[0]["ndcg_at_k"]) == pytest.approx(0.6)
    assert rows[1]["model_name"] == "popularity"


def test_write_model_comparison_rejects_empty_results(
    tmp_path: Path,
) -> None:
    with pytest.raises(
        ValueError,
        match="results must not be empty",
    ):
        write_model_comparison(
            {},
            tmp_path / "comparison.csv",
        )
