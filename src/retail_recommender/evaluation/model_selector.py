"""Utilities for selecting the best evaluated recommender model."""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import pandas as pd


class ModelSelectionError(ValueError):
    """Raised when a model cannot be selected from an evaluation report."""


@dataclass(frozen=True)
class ModelSelection:
    """Structured result of the model selection process."""

    model_name: str
    primary_metric: str
    primary_metric_value: float
    k: int
    evaluated_users: int
    metrics: dict[str, float]

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable representation of the selection."""

        return asdict(self)


def load_model_comparison(path: Path) -> pd.DataFrame:
    """Load the model comparison report from a CSV file."""

    if not path.exists():
        message = f"Model comparison file does not exist: {path}"
        raise FileNotFoundError(message)

    comparison = pd.read_csv(path)

    if comparison.empty:
        message = f"Model comparison file is empty: {path}"
        raise ModelSelectionError(message)

    return comparison


def select_best_model(
    comparison: pd.DataFrame,
    primary_metric: str,
    tie_breakers: Sequence[str] = (),
) -> ModelSelection:
    """Select the best model using a primary metric and deterministic tie-breakers."""

    if comparison.empty:
        raise ModelSelectionError("Model comparison data cannot be empty.")

    required_columns = {
        "model_name",
        "k",
        "evaluated_users",
        primary_metric,
        *tie_breakers,
    }
    missing_columns = sorted(required_columns.difference(comparison.columns))

    if missing_columns:
        missing = ", ".join(missing_columns)
        message = f"Model comparison is missing required columns: {missing}"
        raise ModelSelectionError(message)

    if comparison["model_name"].isna().any():
        raise ModelSelectionError("Column 'model_name' contains null values.")

    metric_columns = [primary_metric, *tie_breakers]
    normalized = comparison.copy()

    for metric in metric_columns:
        normalized[metric] = pd.to_numeric(normalized[metric], errors="coerce")

        if normalized[metric].isna().any():
            message = f"Metric column '{metric}' contains invalid or null values."
            raise ModelSelectionError(message)

    normalized["k"] = pd.to_numeric(normalized["k"], errors="coerce")
    normalized["evaluated_users"] = pd.to_numeric(
        normalized["evaluated_users"],
        errors="coerce",
    )

    if normalized[["k", "evaluated_users"]].isna().any().any():
        raise ModelSelectionError(
            "Columns 'k' and 'evaluated_users' must contain valid numbers."
        )

    sort_columns = [primary_metric, *tie_breakers, "model_name"]
    ascending = [False] * len(metric_columns) + [True]

    ordered = normalized.sort_values(
        by=sort_columns,
        ascending=ascending,
        kind="stable",
    )
    winner = ordered.iloc[0]

    metrics = {
        column: float(winner[column])
        for column in comparison.columns
        if column.endswith("_at_k")
    }

    return ModelSelection(
        model_name=str(winner["model_name"]),
        primary_metric=primary_metric,
        primary_metric_value=float(winner[primary_metric]),
        k=int(winner["k"]),
        evaluated_users=int(winner["evaluated_users"]),
        metrics=metrics,
    )


def save_model_selection(
    selection: ModelSelection,
    output_path: Path,
) -> None:
    """Persist the selected model metadata as formatted JSON."""

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(selection.to_dict(), indent=2, sort_keys=True),
        encoding="utf-8",
    )
