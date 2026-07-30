"""Report writers for model evaluation results."""

from __future__ import annotations

import csv
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from retail_recommender.evaluation.evaluator import EvaluationResult

COMPARISON_COLUMNS = [
    "model_name",
    "k",
    "evaluated_users",
    "precision_at_k",
    "recall_at_k",
    "ndcg_at_k",
    "map_at_k",
    "coverage_at_k",
]


def write_evaluation_report(
    result: EvaluationResult,
    path: Path,
    *,
    model_name: str,
    metadata: Mapping[str, Any] | None = None,
) -> None:
    """Write one model evaluation result as JSON.

    Args:
        result: Aggregated evaluation result.
        path: Destination JSON path.
        model_name: Evaluated model name.
        metadata: Optional additional report metadata.
    """
    if not model_name.strip():
        msg = "model_name must not be empty"
        raise ValueError(msg)

    path.parent.mkdir(parents=True, exist_ok=True)

    payload: dict[str, Any] = {
        "model_name": model_name,
        "metrics": result.to_dict(),
    }

    if metadata:
        payload["metadata"] = dict(metadata)

    path.write_text(
        json.dumps(
            payload,
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )


def write_model_comparison(
    results: Mapping[str, EvaluationResult],
    path: Path,
) -> None:
    """Write comparable model metrics as CSV.

    Args:
        results: Evaluation results keyed by model name.
        path: Destination CSV path.

    Raises:
        ValueError: If no model results are supplied.
    """
    if not results:
        msg = "results must not be empty"
        raise ValueError(msg)

    path.parent.mkdir(parents=True, exist_ok=True)

    rows = [
        _build_comparison_row(
            model_name=model_name,
            result=result,
        )
        for model_name, result in results.items()
    ]

    rows.sort(
        key=lambda row: (
            -float(row["ndcg_at_k"]),
            str(row["model_name"]),
        )
    )

    with path.open(
        "w",
        encoding="utf-8",
        newline="",
    ) as comparison_file:
        writer = csv.DictWriter(
            comparison_file,
            fieldnames=COMPARISON_COLUMNS,
        )
        writer.writeheader()
        writer.writerows(rows)


def _build_comparison_row(
    *,
    model_name: str,
    result: EvaluationResult,
) -> dict[str, str | int | float]:
    """Convert one result into a flat comparison row."""
    return {
        "model_name": model_name,
        **result.to_dict(),
    }
