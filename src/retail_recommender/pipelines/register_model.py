"""Register the best evaluated recommender in MLflow Model Registry."""

from __future__ import annotations

import json
import logging
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import mlflow
import pandas as pd
import yaml
from mlflow.models import infer_signature

from retail_recommender.config.settings import get_settings
from retail_recommender.tracking.item_knn_pyfunc import (
    ItemKNNPyfuncModel,
)
from retail_recommender.tracking.registry import (
    ModelRegistry,
    RegisteredVersion,
)

LOGGER = logging.getLogger(__name__)

PARAMS_PATH = Path("params.yaml")
DATA_CONFIG_PATH = Path("configs/data.yaml")
EVALUATION_CONFIG_PATH = Path("configs/evaluation.yaml")
REGISTRY_CONFIG_PATH = Path("configs/registry.yaml")

SUPPORTED_MODEL = "item_knn"
MODEL_ARTIFACT_NAME = "selected-recommender"
REGISTER_RUN_NAME = "register_selected_model"


@dataclass(frozen=True)
class RegistrationPaths:
    """Filesystem paths required by model registration."""

    train_interactions: Path
    selected_model: Path
    registry_report: Path


@dataclass(frozen=True)
class RegistrationResult:
    """Structured result of a completed model registration."""

    registered_model_name: str
    version: str
    alias: str
    source_model: str
    model_uri: str
    run_id: str
    primary_metric: str
    primary_metric_value: float
    k: int
    evaluated_users: int

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable representation."""

        return asdict(self)


def run_model_registration(
    *,
    promote_to_production: bool = False,
) -> RegistrationResult:
    """Log and register the selected recommender model."""

    settings = get_settings()
    paths = _load_registration_paths()

    params = _load_yaml(PARAMS_PATH)
    item_knn_params = _get_mapping(params, "item_knn")
    evaluation_params = _get_mapping(params, "evaluation")

    selection = _load_json(paths.selected_model)
    selected_model = _get_required_string(
        selection,
        "model_name",
    )

    if selected_model != SUPPORTED_MODEL:
        message = (
            f"Selected model '{selected_model}' is not supported by "
            f"this registration pipeline. Supported model: "
            f"'{SUPPORTED_MODEL}'."
        )
        raise ValueError(message)

    n_neighbors = _get_required_int(
        item_knn_params,
        "n_neighbors",
    )
    minimum_similarity = _get_required_float(
        item_knn_params,
        "minimum_similarity",
    )
    k = _get_required_int(
        evaluation_params,
        "k",
    )

    train_interactions = pd.read_parquet(paths.train_interactions)
    _validate_train_interactions(train_interactions)

    input_example = pd.DataFrame(
        {
            "user_idx": train_interactions["user_idx"]
            .drop_duplicates()
            .head(2)
            .astype("Int64")
            .tolist()
        }
    ).astype({"user_idx": "Int64"})

    pyfunc_model = ItemKNNPyfuncModel(
        n_neighbors=n_neighbors,
        minimum_similarity=minimum_similarity,
        default_k=k,
    )

    mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
    mlflow.set_registry_uri(settings.mlflow_tracking_uri)
    mlflow.set_experiment(settings.mlflow_experiment_name)

    registry = ModelRegistry(settings=settings)
    registry.ensure_registered_model(
        description=(
            "Best RetailRocket Top-K recommender selected from "
            "Popularity, Item-KNN and Neural CF."
        )
    )

    with mlflow.start_run(
        run_name=REGISTER_RUN_NAME,
        tags={
            "pipeline": "register_model",
            "selected_model": selected_model,
        },
    ) as run:
        mlflow.log_params(
            {
                "selected_model": selected_model,
                "n_neighbors": n_neighbors,
                "minimum_similarity": minimum_similarity,
                "k": k,
                "primary_metric": selection["primary_metric"],
            }
        )
        mlflow.log_metrics(
            {
                "selected_primary_metric_value": float(
                    selection["primary_metric_value"]
                ),
            }
        )
        mlflow.log_artifact(
            str(paths.selected_model),
            artifact_path="selection",
        )

        model_output_example = pd.DataFrame(
            {
                "user_idx": input_example["user_idx"],
                "recommendations": ["[]"] * len(input_example),
            }
        )
        signature = infer_signature(
            input_example,
            model_output_example,
        )

        model_info = mlflow.pyfunc.log_model(
            name=MODEL_ARTIFACT_NAME,
            python_model=pyfunc_model,
            artifacts={
                "train_interactions": str(paths.train_interactions),
            },
            input_example=input_example,
            signature=signature,
            metadata={
                "selected_model": selected_model,
                "primary_metric": selection["primary_metric"],
            },
        )

        run_id = run.info.run_id
        model_uri = model_info.model_uri

    version = registry.create_version(
        model_uri=model_uri,
        run_id=run_id,
        description=(
            "Item-KNN selected automatically by " f"{selection['primary_metric']}."
        ),
        tags=_build_version_tags(selection),
    )
    registry.promote_to_staging(version.version)

    alias = settings.mlflow_staging_alias

    if promote_to_production:
        registry.promote_to_production(version.version)
        alias = settings.mlflow_production_alias

    result = _build_registration_result(
        registered_version=version,
        alias=alias,
        source_model=selected_model,
        model_uri=model_uri,
        run_id=run_id,
        selection=selection,
    )
    _write_registration_report(
        result=result,
        path=paths.registry_report,
    )

    LOGGER.info(
        "Registered model '%s' version %s with alias '%s'.",
        result.registered_model_name,
        result.version,
        result.alias,
    )

    return result


def _load_registration_paths() -> RegistrationPaths:
    """Load paths required for model registration."""

    data_config = _load_yaml(DATA_CONFIG_PATH)
    evaluation_config = _get_mapping(
        _load_yaml(EVALUATION_CONFIG_PATH),
        "evaluation",
    )
    registry_config = _get_mapping(
        _load_yaml(REGISTRY_CONFIG_PATH),
        "registry",
    )

    return RegistrationPaths(
        train_interactions=Path(
            _get_required_string(
                data_config,
                "train_positive_path",
            )
        ),
        selected_model=Path(
            _get_required_string(
                evaluation_config,
                "selected_model_path",
            )
        ),
        registry_report=Path(
            _get_required_string(
                registry_config,
                "report_path",
            )
        ),
    )


def _build_version_tags(
    selection: Mapping[str, Any],
) -> dict[str, str]:
    """Build MLflow tags from the selected-model report."""

    return {
        "model_type": _get_required_string(
            selection,
            "model_name",
        ),
        "primary_metric": _get_required_string(
            selection,
            "primary_metric",
        ),
        "primary_metric_value": str(
            _get_required_float(
                selection,
                "primary_metric_value",
            )
        ),
        "k": str(_get_required_int(selection, "k")),
        "evaluated_users": str(
            _get_required_int(
                selection,
                "evaluated_users",
            )
        ),
        "validation_status": "passed",
    }


def _build_registration_result(
    *,
    registered_version: RegisteredVersion,
    alias: str,
    source_model: str,
    model_uri: str,
    run_id: str,
    selection: Mapping[str, Any],
) -> RegistrationResult:
    """Build the final model registration result."""

    return RegistrationResult(
        registered_model_name=registered_version.model_name,
        version=registered_version.version,
        alias=alias,
        source_model=source_model,
        model_uri=model_uri,
        run_id=run_id,
        primary_metric=_get_required_string(
            selection,
            "primary_metric",
        ),
        primary_metric_value=_get_required_float(
            selection,
            "primary_metric_value",
        ),
        k=_get_required_int(selection, "k"),
        evaluated_users=_get_required_int(
            selection,
            "evaluated_users",
        ),
    )


def _validate_train_interactions(
    interactions: pd.DataFrame,
) -> None:
    """Validate columns required to rebuild Item-KNN."""

    required_columns = {
        "user_idx",
        "item_idx",
        "interaction_score",
        "interaction_count",
        "target",
    }
    missing_columns = sorted(required_columns.difference(interactions.columns))

    if missing_columns:
        missing = ", ".join(missing_columns)
        message = "Training interactions are missing columns: " f"{missing}"
        raise ValueError(message)

    if interactions.empty:
        raise ValueError("Training interactions cannot be empty.")


def _write_registration_report(
    *,
    result: RegistrationResult,
    path: Path,
) -> None:
    """Persist model registration metadata."""

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            result.to_dict(),
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )


def _load_yaml(path: Path) -> dict[str, Any]:
    """Load a YAML mapping."""

    with path.open(encoding="utf-8") as file:
        data = yaml.safe_load(file)

    if not isinstance(data, dict):
        raise ValueError(f"Configuration file must contain a mapping: {path}")

    return data


def _load_json(path: Path) -> dict[str, Any]:
    """Load a JSON mapping."""

    if not path.exists():
        raise FileNotFoundError(f"Required JSON file does not exist: {path}")

    data = json.loads(path.read_text(encoding="utf-8"))

    if not isinstance(data, dict):
        raise ValueError(f"JSON file must contain an object: {path}")

    return data


def _get_mapping(
    mapping: Mapping[str, Any],
    key: str,
) -> Mapping[str, Any]:
    """Return a required nested mapping."""

    value = mapping.get(key)

    if not isinstance(value, dict):
        raise ValueError(f"Configuration key '{key}' must be a mapping.")

    return value


def _get_required_string(
    mapping: Mapping[str, Any],
    key: str,
) -> str:
    """Return a required non-empty string."""

    value = mapping.get(key)

    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"Configuration key '{key}' must be a non-empty string.")

    return value.strip()


def _get_required_int(
    mapping: Mapping[str, Any],
    key: str,
) -> int:
    """Return a required integer."""

    value = mapping.get(key)

    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"Configuration key '{key}' must be an integer.")

    return value


def _get_required_float(
    mapping: Mapping[str, Any],
    key: str,
) -> float:
    """Return a required numeric value."""

    value = mapping.get(key)

    if isinstance(value, bool) or not isinstance(
        value,
        int | float,
    ):
        raise ValueError(f"Configuration key '{key}' must be numeric.")

    return float(value)


def main() -> None:
    """Run model registration from the command line."""

    result = run_model_registration()
    print(
        "Registered "
        f"{result.registered_model_name} "
        f"version {result.version} "
        f"with alias '{result.alias}'."
    )


if __name__ == "__main__":
    main()
