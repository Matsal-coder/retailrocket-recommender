"""Tests for model selection utilities."""

import json
from pathlib import Path

import pandas as pd
import pytest

from retail_recommender.evaluation.model_selector import (
    ModelSelection,
    ModelSelectionError,
    load_model_comparison,
    save_model_selection,
    select_best_model,
)

PRIMARY_METRIC = "ndcg_at_k"
TIE_BREAKERS = ("recall_at_k", "map_at_k", "coverage_at_k")
DEFAULT_K = 10
DEFAULT_USERS = 50


def build_comparison() -> pd.DataFrame:
    """Create a valid model comparison fixture."""

    return pd.DataFrame(
        [
            {
                "model_name": "item_knn",
                "k": 10,
                "evaluated_users": 50,
                "precision_at_k": 0.006,
                "recall_at_k": 0.050,
                "ndcg_at_k": 0.040,
                "map_at_k": 0.034,
                "coverage_at_k": 0.005,
            },
            {
                "model_name": "popularity",
                "k": 10,
                "evaluated_users": 50,
                "precision_at_k": 0.006,
                "recall_at_k": 0.035,
                "ndcg_at_k": 0.023,
                "map_at_k": 0.015,
                "coverage_at_k": 0.001,
            },
        ]
    )


def test_load_model_comparison_reads_csv(tmp_path: Path) -> None:
    comparison_path = tmp_path / "model_comparison.csv"
    expected = build_comparison()
    expected.to_csv(comparison_path, index=False)

    result = load_model_comparison(comparison_path)

    pd.testing.assert_frame_equal(result, expected)


def test_load_model_comparison_rejects_missing_file(tmp_path: Path) -> None:
    comparison_path = tmp_path / "missing.csv"

    with pytest.raises(FileNotFoundError, match="does not exist"):
        load_model_comparison(comparison_path)


def test_load_model_comparison_rejects_empty_csv(tmp_path: Path) -> None:
    comparison_path = tmp_path / "empty.csv"
    comparison_path.write_text("model_name,ndcg_at_k\n", encoding="utf-8")

    with pytest.raises(ModelSelectionError, match="is empty"):
        load_model_comparison(comparison_path)


def test_select_best_model_uses_primary_metric() -> None:
    comparison = build_comparison()

    selection = select_best_model(
        comparison=comparison,
        primary_metric=PRIMARY_METRIC,
        tie_breakers=TIE_BREAKERS,
    )

    assert selection.model_name == "item_knn"
    assert selection.primary_metric == PRIMARY_METRIC
    assert selection.primary_metric_value == pytest.approx(0.040)
    assert selection.k == DEFAULT_K
    assert selection.evaluated_users == DEFAULT_USERS


def test_select_best_model_uses_tie_breaker() -> None:
    comparison = build_comparison()
    comparison.loc[:, PRIMARY_METRIC] = 0.040
    comparison.loc[0, "recall_at_k"] = 0.030
    comparison.loc[1, "recall_at_k"] = 0.050

    selection = select_best_model(
        comparison=comparison,
        primary_metric=PRIMARY_METRIC,
        tie_breakers=TIE_BREAKERS,
    )

    assert selection.model_name == "popularity"


def test_select_best_model_uses_model_name_after_full_tie() -> None:
    comparison = build_comparison()

    for metric in (PRIMARY_METRIC, *TIE_BREAKERS):
        comparison.loc[:, metric] = 0.050

    selection = select_best_model(
        comparison=comparison,
        primary_metric=PRIMARY_METRIC,
        tie_breakers=TIE_BREAKERS,
    )

    assert selection.model_name == "item_knn"


def test_select_best_model_rejects_empty_dataframe() -> None:
    with pytest.raises(ModelSelectionError, match="cannot be empty"):
        select_best_model(
            comparison=pd.DataFrame(),
            primary_metric=PRIMARY_METRIC,
            tie_breakers=TIE_BREAKERS,
        )


def test_select_best_model_rejects_missing_metric() -> None:
    comparison = build_comparison().drop(columns=[PRIMARY_METRIC])

    with pytest.raises(ModelSelectionError, match=PRIMARY_METRIC):
        select_best_model(
            comparison=comparison,
            primary_metric=PRIMARY_METRIC,
            tie_breakers=TIE_BREAKERS,
        )


def test_select_best_model_rejects_invalid_metric_value() -> None:
    comparison = build_comparison()
    comparison[PRIMARY_METRIC] = comparison[PRIMARY_METRIC].astype(object)
    comparison.loc[0, PRIMARY_METRIC] = "invalid"

    with pytest.raises(ModelSelectionError, match="invalid or null"):
        select_best_model(
            comparison=comparison,
            primary_metric=PRIMARY_METRIC,
            tie_breakers=TIE_BREAKERS,
        )


def test_save_model_selection_writes_json(tmp_path: Path) -> None:
    output_path = tmp_path / "reports" / "selected_model.json"
    selection = ModelSelection(
        model_name="item_knn",
        primary_metric=PRIMARY_METRIC,
        primary_metric_value=0.040,
        k=10,
        evaluated_users=50,
        metrics={
            "precision_at_k": 0.006,
            "recall_at_k": 0.050,
            "ndcg_at_k": 0.040,
            "map_at_k": 0.034,
            "coverage_at_k": 0.005,
        },
    )

    save_model_selection(selection, output_path)

    saved = json.loads(output_path.read_text(encoding="utf-8"))

    assert saved["model_name"] == "item_knn"
    assert saved["primary_metric"] == PRIMARY_METRIC
    assert saved["primary_metric_value"] == pytest.approx(0.040)
    assert saved["k"] == DEFAULT_K


def test_select_best_model_rejects_null_model_name() -> None:
    comparison = build_comparison()
    comparison.loc[0, "model_name"] = None

    with pytest.raises(ModelSelectionError, match="model_name"):
        select_best_model(
            comparison=comparison,
            primary_metric=PRIMARY_METRIC,
            tie_breakers=TIE_BREAKERS,
        )


@pytest.mark.parametrize("column", ["k", "evaluated_users"])
def test_select_best_model_rejects_invalid_metadata(column: str) -> None:
    comparison = build_comparison()
    comparison[column] = comparison[column].astype(object)
    comparison.loc[0, column] = "invalid"

    with pytest.raises(
        ModelSelectionError,
        match="must contain valid numbers",
    ):
        select_best_model(
            comparison=comparison,
            primary_metric=PRIMARY_METRIC,
            tie_breakers=TIE_BREAKERS,
        )
