"""Evaluate recommender models with Top-K ranking metrics."""

from __future__ import annotations

import json
import logging
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import pandas as pd
import torch
import yaml

from retail_recommender.evaluation.evaluator import (
    EvaluationResult,
    RecommenderEvaluator,
)
from retail_recommender.evaluation.neural_recommender import (
    NeuralCFRecommender,
)
from retail_recommender.evaluation.reports import (
    write_evaluation_report,
    write_model_comparison,
)
from retail_recommender.models.item_knn import (
    ItemKNNRecommender,
)
from retail_recommender.models.neural_cf import (
    NeuralCollaborativeFiltering,
)
from retail_recommender.models.popularity import (
    PopularityRecommender,
)
from retail_recommender.tracking.mlflow_tracker import (
    MLflowTracker,
)
from retail_recommender.training.seed import set_global_seed

LOGGER = logging.getLogger(__name__)

PARAMS_PATH = Path("params.yaml")
EVALUATION_CONFIG_PATH = Path("configs/evaluation.yaml")

BASELINE_REQUIRED_COLUMNS = {
    "user_idx",
    "item_idx",
    "interaction_score",
    "interaction_count",
    "target",
}

TEST_REQUIRED_COLUMNS = {
    "user_idx",
    "item_idx",
    "target",
}

MODEL_NAMES = (
    "popularity",
    "item_knn",
    "neural_cf",
)


def run_evaluation() -> dict[str, EvaluationResult]:
    """Evaluate all project recommender models."""
    params = _load_yaml(PARAMS_PATH)
    evaluation_config = _load_yaml(EVALUATION_CONFIG_PATH)

    training_params = _get_mapping(params, "training")
    model_params = _get_mapping(params, "model")
    evaluation_params = _get_mapping(params, "evaluation")
    item_knn_params = _get_mapping(params, "item_knn")
    evaluation_paths = _get_mapping(
        evaluation_config,
        "evaluation",
    )

    random_seed = _get_required_int(
        training_params,
        "random_seed",
    )
    set_global_seed(random_seed)

    train_interactions_path = Path(
        _get_required_string(
            evaluation_paths,
            "train_interactions_path",
        )
    )
    test_path = Path(
        _get_required_string(
            evaluation_paths,
            "test_data_path",
        )
    )
    checkpoint_path = Path(
        _get_required_string(
            evaluation_paths,
            "checkpoint_path",
        )
    )
    training_report_path = Path(
        _get_required_string(
            evaluation_paths,
            "training_report_path",
        )
    )
    output_directory = Path(
        _get_required_string(
            evaluation_paths,
            "output_directory",
        )
    )

    train_interactions = pd.read_parquet(train_interactions_path)
    test_interactions = pd.read_parquet(test_path)

    _validate_interactions(
        train_interactions,
        split_name="training",
        required_columns=BASELINE_REQUIRED_COLUMNS,
    )
    _validate_interactions(
        test_interactions,
        split_name="test",
        required_columns=TEST_REQUIRED_COLUMNS,
    )

    positive_test = _positive_interactions(test_interactions)
    ground_truth = _build_user_item_mapping(positive_test)

    exclude_seen_items = _get_optional_bool(
        evaluation_params,
        "exclude_seen_items",
        default=True,
    )
    seen_items = (
        _build_user_item_mapping(train_interactions) if exclude_seen_items else {}
    )

    catalog_items = sorted(
        set(train_interactions["item_idx"]) | set(test_interactions["item_idx"])
    )

    k = _get_required_int(
        evaluation_params,
        "k",
    )
    maximum_users = _get_optional_int(
        evaluation_params,
        "maximum_users",
    )
    num_users, num_items = _load_model_dimensions(training_report_path)

    models = _build_models(
        train_interactions=train_interactions,
        model_params=model_params,
        item_knn_params=item_knn_params,
        evaluation_params=evaluation_params,
        checkpoint_path=checkpoint_path,
        num_users=num_users,
        num_items=num_items,
    )

    results: dict[str, EvaluationResult] = {}
    tracker = MLflowTracker()

    output_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    for model_name, recommender in models.items():
        LOGGER.info(
            "Evaluating model: %s",
            model_name,
        )

        evaluator = RecommenderEvaluator(
            recommender=recommender,
            k=k,
            catalog_items=catalog_items,
            maximum_users=maximum_users,
        )

        with tracker.run(
            run_name=f"{model_name}_evaluation",
            tags={
                "model_type": model_name,
                "pipeline": "evaluate",
            },
        ):
            result = evaluator.evaluate(
                relevant_items_by_user=ground_truth,
                seen_items_by_user=seen_items,
            )
            results[model_name] = result

            tracker.log_parameters(
                {
                    "model_name": model_name,
                    "k": k,
                    "maximum_users": (
                        maximum_users if maximum_users is not None else "all"
                    ),
                    "exclude_seen_items": exclude_seen_items,
                    "random_seed": random_seed,
                }
            )
            tracker.log_metrics(result.to_dict())

            report_path = output_directory / f"{model_name}_metrics.json"
            write_evaluation_report(
                result=result,
                path=report_path,
                model_name=model_name,
            )
            tracker.log_artifact(
                report_path,
                artifact_directory="evaluation",
            )

    comparison_path = output_directory / "model_comparison.csv"
    write_model_comparison(
        results=results,
        path=comparison_path,
    )

    with tracker.run(
        run_name="model_comparison",
        tags={
            "pipeline": "evaluate",
            "artifact_type": "comparison",
        },
    ):
        tracker.log_parameters(
            {
                "models": list(MODEL_NAMES),
                "k": k,
                "random_seed": random_seed,
            }
        )
        tracker.log_artifact(
            comparison_path,
            artifact_directory="evaluation",
        )

    return results


def _build_models(
    *,
    train_interactions: pd.DataFrame,
    model_params: Mapping[str, Any],
    item_knn_params: Mapping[str, Any],
    evaluation_params: Mapping[str, Any],
    checkpoint_path: Path,
    num_users: int,
    num_items: int,
) -> dict[str, Any]:
    """Build and fit all recommender implementations."""
    popularity = PopularityRecommender()
    popularity.fit(train_interactions)

    item_knn = ItemKNNRecommender(
        n_neighbors=_get_required_int(
            item_knn_params,
            "n_neighbors",
        ),
        minimum_similarity=_get_required_float(
            item_knn_params,
            "minimum_similarity",
        ),
    )
    item_knn.fit(train_interactions)

    neural_model = NeuralCollaborativeFiltering(
        num_users=num_users,
        num_items=num_items,
        embedding_dim=_get_required_int(
            model_params,
            "embedding_dim",
        ),
        hidden_layers=_get_required_int_sequence(
            model_params,
            "hidden_layers",
        ),
        dropout=_get_required_float(
            model_params,
            "dropout",
        ),
    )

    if not checkpoint_path.exists():
        msg = "Neural CF checkpoint not found: " f"{checkpoint_path}"
        raise FileNotFoundError(msg)

    device = _resolve_evaluation_device()

    state_dict = torch.load(
        checkpoint_path,
        map_location=device,
        weights_only=True,
    )
    neural_model.load_state_dict(state_dict)

    neural_recommender = NeuralCFRecommender(
        model=neural_model,
        num_items=num_items,
        device=device,
        candidate_batch_size=_get_required_int(
            evaluation_params,
            "candidate_batch_size",
        ),
    )

    return {
        "popularity": popularity,
        "item_knn": item_knn,
        "neural_cf": neural_recommender,
    }


def _positive_interactions(
    interactions: pd.DataFrame,
) -> pd.DataFrame:
    """Keep positive implicit-feedback interactions."""
    return interactions.loc[interactions["target"] > 0].copy()


def _build_user_item_mapping(
    interactions: pd.DataFrame,
) -> dict[int, set[int]]:
    """Build a mapping from encoded user IDs to item sets."""
    grouped_items = interactions.groupby(
        "user_idx",
        sort=True,
    )["item_idx"]

    return {
        int(user_id): {int(item_id) for item_id in user_items}
        for user_id, user_items in grouped_items
    }


def _load_model_dimensions(
    report_path: Path,
) -> tuple[int, int]:
    """Load neural embedding dimensions from the training report."""
    if not report_path.exists():
        msg = f"Training report not found: {report_path}"
        raise FileNotFoundError(msg)

    report = json.loads(report_path.read_text(encoding="utf-8"))
    model_metadata = report.get("model")

    if not isinstance(model_metadata, dict):
        msg = "Training report is missing model metadata"
        raise ValueError(msg)

    num_users = model_metadata.get("num_users")
    num_items = model_metadata.get("num_items")

    if not isinstance(num_users, int) or isinstance(num_users, bool) or num_users <= 0:
        msg = "Training report contains invalid num_users"
        raise ValueError(msg)

    if not isinstance(num_items, int) or isinstance(num_items, bool) or num_items <= 0:
        msg = "Training report contains invalid num_items"
        raise ValueError(msg)

    return num_users, num_items


def _resolve_entity_counts(
    train_interactions: pd.DataFrame,
    test_interactions: pd.DataFrame,
) -> tuple[int, int]:
    """Resolve entity counts from dataframes.

    Retained as a helper for validation and backward-compatible unit tests.
    The production evaluation pipeline uses the dimensions persisted by
    training to guarantee checkpoint compatibility.
    """
    maximum_user = max(
        int(train_interactions["user_idx"].max()),
        int(test_interactions["user_idx"].max()),
    )
    maximum_item = max(
        int(train_interactions["item_idx"].max()),
        int(test_interactions["item_idx"].max()),
    )

    return maximum_user + 1, maximum_item + 1


def _validate_interactions(
    interactions: pd.DataFrame,
    *,
    split_name: str,
    required_columns: set[str],
) -> None:
    """Validate one evaluation interaction dataset."""
    missing_columns = required_columns.difference(interactions.columns)

    if missing_columns:
        formatted_columns = ", ".join(sorted(missing_columns))
        msg = f"{split_name} data is missing columns: " f"{formatted_columns}"
        raise ValueError(msg)

    if interactions.empty:
        msg = f"{split_name} data must not be empty"
        raise ValueError(msg)


def _resolve_evaluation_device() -> torch.device:
    """Resolve the available inference device."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _load_yaml(path: Path) -> dict[str, Any]:
    """Load a YAML configuration mapping."""
    if not path.exists():
        msg = f"Configuration file not found: {path}"
        raise FileNotFoundError(msg)

    loaded_content = yaml.safe_load(path.read_text(encoding="utf-8"))

    if not isinstance(loaded_content, dict):
        msg = f"Configuration must be a mapping: {path}"
        raise ValueError(msg)

    return loaded_content


def _get_mapping(
    values: Mapping[str, Any],
    key: str,
) -> Mapping[str, Any]:
    """Read a required nested mapping."""
    nested_values = values.get(key)

    if not isinstance(nested_values, dict):
        msg = f"Missing configuration section: {key}"
        raise ValueError(msg)

    return nested_values


def _get_required_string(
    values: Mapping[str, Any],
    key: str,
) -> str:
    """Read a required non-empty string."""
    value = values.get(key)

    if not isinstance(value, str) or not value.strip():
        msg = "Configuration value must be a string: " f"{key}"
        raise ValueError(msg)

    return value.strip()


def _get_required_int(
    values: Mapping[str, Any],
    key: str,
) -> int:
    """Read a required integer."""
    value = values.get(key)

    if not isinstance(value, int) or isinstance(value, bool):
        msg = "Configuration value must be an integer: " f"{key}"
        raise ValueError(msg)

    return value


def _get_optional_int(
    values: Mapping[str, Any],
    key: str,
) -> int | None:
    """Read an optional integer."""
    value = values.get(key)

    if value is None:
        return None

    if not isinstance(value, int) or isinstance(value, bool):
        msg = "Configuration value must be an integer: " f"{key}"
        raise ValueError(msg)

    return value


def _get_optional_bool(
    values: Mapping[str, Any],
    key: str,
    *,
    default: bool,
) -> bool:
    """Read an optional boolean value."""
    value = values.get(key, default)

    if not isinstance(value, bool):
        msg = "Configuration value must be a boolean: " f"{key}"
        raise ValueError(msg)

    return value


def _get_required_float(
    values: Mapping[str, Any],
    key: str,
) -> float:
    """Read a required numerical value."""
    value = values.get(key)

    if not isinstance(value, int | float) or isinstance(value, bool):
        msg = "Configuration value must be numeric: " f"{key}"
        raise ValueError(msg)

    return float(value)


def _get_required_int_sequence(
    values: Mapping[str, Any],
    key: str,
) -> tuple[int, ...]:
    """Read a non-empty integer sequence."""
    value = values.get(key)

    if not isinstance(value, list | tuple) or not value:
        msg = "Configuration value must be a non-empty " f"integer sequence: {key}"
        raise ValueError(msg)

    if any(not isinstance(item, int) or isinstance(item, bool) for item in value):
        msg = "Configuration value must contain integers: " f"{key}"
        raise ValueError(msg)

    return tuple(value)


def main() -> None:
    """Execute the evaluation pipeline."""
    run_evaluation()


if __name__ == "__main__":
    main()
