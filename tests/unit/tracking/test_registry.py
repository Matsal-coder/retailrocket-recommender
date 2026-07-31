"""Unit tests for the MLflow Model Registry integration."""

from unittest.mock import MagicMock

import pytest
from mlflow.entities.model_registry import ModelVersion
from mlflow.exceptions import MlflowException
from mlflow.protos.databricks_pb2 import RESOURCE_DOES_NOT_EXIST

from retail_recommender.config.settings import Settings
from retail_recommender.tracking.registry import (
    ModelRegistry,
    ModelRegistryError,
    _normalize_alias,
    _normalize_tags,
    _normalize_version,
)

MODEL_NAME = "RetailRocketRecommender"
MODEL_VERSION = "3"
MODEL_URI = "runs:/run-123/item-knn-model"
RUN_ID = "run-123"
DEFAULT_CALL_COUNT = 2


@pytest.fixture
def settings() -> Settings:
    """Create registry settings for unit tests."""

    return Settings(
        mlflow_tracking_uri="sqlite:///test-mlflow.db",
        mlflow_registered_model_name=MODEL_NAME,
        mlflow_staging_alias="staging",
        mlflow_production_alias="production",
    )


@pytest.fixture
def client() -> MagicMock:
    """Create a mocked MLflow client."""

    return MagicMock()


@pytest.fixture
def registry(
    settings: Settings,
    client: MagicMock,
) -> ModelRegistry:
    """Create a registry backed by a mocked client."""

    return ModelRegistry(
        settings=settings,
        client=client,
    )


def test_model_name_comes_from_settings(
    registry: ModelRegistry,
) -> None:
    assert registry.model_name == MODEL_NAME


def test_ensure_registered_model_keeps_existing_model(
    registry: ModelRegistry,
    client: MagicMock,
) -> None:
    client.get_registered_model.return_value = MagicMock()

    registry.ensure_registered_model()

    client.get_registered_model.assert_called_once_with(MODEL_NAME)
    client.create_registered_model.assert_not_called()


def test_ensure_registered_model_updates_description(
    registry: ModelRegistry,
    client: MagicMock,
) -> None:
    client.get_registered_model.return_value = MagicMock()

    registry.ensure_registered_model(
        description="RetailRocket Top-K recommender.",
    )

    client.update_registered_model.assert_called_once_with(
        name=MODEL_NAME,
        description="RetailRocket Top-K recommender.",
    )


def test_ensure_registered_model_creates_missing_model(
    registry: ModelRegistry,
    client: MagicMock,
) -> None:
    client.get_registered_model.side_effect = MlflowException(
        "Missing model.",
        error_code=RESOURCE_DOES_NOT_EXIST,
    )

    registry.ensure_registered_model(
        description="RetailRocket Top-K recommender.",
    )

    client.create_registered_model.assert_called_once_with(
        name=MODEL_NAME,
        description="RetailRocket Top-K recommender.",
    )


def test_ensure_registered_model_wraps_creation_error(
    registry: ModelRegistry,
    client: MagicMock,
) -> None:
    client.get_registered_model.side_effect = MlflowException(
        "Missing model.",
        error_code=RESOURCE_DOES_NOT_EXIST,
    )
    client.create_registered_model.side_effect = MlflowException(
        "Creation failed.",
    )

    with pytest.raises(
        ModelRegistryError,
        match="Could not create registered model",
    ):
        registry.ensure_registered_model()


def test_create_version_returns_structured_metadata(
    registry: ModelRegistry,
    client: MagicMock,
) -> None:
    client.create_model_version.return_value = MagicMock(
        version=MODEL_VERSION,
    )

    result = registry.create_version(
        model_uri=MODEL_URI,
        run_id=RUN_ID,
        description="Selected Item-KNN model.",
        tags={
            "model_type": "item_knn",
            "primary_metric": "ndcg_at_k",
        },
    )

    client.create_model_version.assert_called_once_with(
        name=MODEL_NAME,
        source=MODEL_URI,
        run_id=RUN_ID,
        description="Selected Item-KNN model.",
        tags={
            "model_type": "item_knn",
            "primary_metric": "ndcg_at_k",
        },
    )
    assert result.model_name == MODEL_NAME
    assert result.version == MODEL_VERSION
    assert result.source == MODEL_URI
    assert result.run_id == RUN_ID


def test_create_version_rejects_empty_uri(
    registry: ModelRegistry,
) -> None:
    with pytest.raises(ValueError, match="URI cannot be empty"):
        registry.create_version(model_uri="  ")


def test_create_version_wraps_mlflow_error(
    registry: ModelRegistry,
    client: MagicMock,
) -> None:
    client.create_model_version.side_effect = MlflowException(
        "Version creation failed.",
    )

    with pytest.raises(
        ModelRegistryError,
        match="Could not create a version",
    ):
        registry.create_version(model_uri=MODEL_URI)


def test_set_version_tags_sets_each_tag(
    registry: ModelRegistry,
    client: MagicMock,
) -> None:
    registry.set_version_tags(
        version=MODEL_VERSION,
        tags={
            "validation_status": "passed",
            "model_type": "item_knn",
        },
    )

    assert client.set_model_version_tag.call_count == DEFAULT_CALL_COUNT
    client.set_model_version_tag.assert_any_call(
        name=MODEL_NAME,
        version=MODEL_VERSION,
        key="validation_status",
        value="passed",
    )
    client.set_model_version_tag.assert_any_call(
        name=MODEL_NAME,
        version=MODEL_VERSION,
        key="model_type",
        value="item_knn",
    )


def test_promote_to_staging_uses_configured_alias(
    registry: ModelRegistry,
    client: MagicMock,
) -> None:
    registry.promote_to_staging(MODEL_VERSION)

    client.set_registered_model_alias.assert_called_once_with(
        name=MODEL_NAME,
        alias="staging",
        version=MODEL_VERSION,
    )


def test_promote_to_production_uses_configured_alias(
    registry: ModelRegistry,
    client: MagicMock,
) -> None:
    registry.promote_to_production(MODEL_VERSION)

    client.set_registered_model_alias.assert_called_once_with(
        name=MODEL_NAME,
        alias="production",
        version=MODEL_VERSION,
    )


def test_set_alias_wraps_mlflow_error(
    registry: ModelRegistry,
    client: MagicMock,
) -> None:
    client.set_registered_model_alias.side_effect = MlflowException(
        "Alias failed.",
    )

    with pytest.raises(
        ModelRegistryError,
        match="Could not assign alias",
    ):
        registry.set_alias(
            version=MODEL_VERSION,
            alias="staging",
        )


def test_get_version_by_alias_returns_model_version(
    registry: ModelRegistry,
    client: MagicMock,
) -> None:
    expected = MagicMock(spec=ModelVersion)
    client.get_model_version_by_alias.return_value = expected

    result = registry.get_version_by_alias("staging")

    assert result is expected
    client.get_model_version_by_alias.assert_called_once_with(
        name=MODEL_NAME,
        alias="staging",
    )


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (" 3 ", "3"),
        (3, "3"),
    ],
)
def test_normalize_version(
    value: object,
    expected: str,
) -> None:
    assert _normalize_version(value) == expected


def test_normalize_version_rejects_empty_value() -> None:
    with pytest.raises(ValueError, match="version cannot be empty"):
        _normalize_version(" ")


def test_normalize_alias_strips_whitespace() -> None:
    assert _normalize_alias(" staging ") == "staging"


def test_normalize_alias_rejects_empty_value() -> None:
    with pytest.raises(ValueError, match="alias cannot be empty"):
        _normalize_alias(" ")


def test_normalize_tags_returns_empty_dictionary_for_none() -> None:
    assert _normalize_tags(None) == {}


def test_normalize_tags_converts_values_to_strings() -> None:
    assert _normalize_tags(
        {
            "model_type": " item_knn ",
            "k": 10,
        }
    ) == {
        "model_type": "item_knn",
        "k": "10",
    }


@pytest.mark.parametrize(
    "tags",
    [
        {"": "item_knn"},
        {"model_type": ""},
    ],
)
def test_normalize_tags_rejects_empty_values(
    tags: dict[str, str],
) -> None:
    with pytest.raises(ValueError):
        _normalize_tags(tags)


def test_ensure_registered_model_wraps_update_error(
    registry: ModelRegistry,
    client: MagicMock,
) -> None:
    client.get_registered_model.return_value = MagicMock()
    client.update_registered_model.side_effect = MlflowException(
        "Update failed.",
    )

    with pytest.raises(
        ModelRegistryError,
        match="Could not update registered model",
    ):
        registry.ensure_registered_model(
            description="Updated description.",
        )


def test_set_version_tags_wraps_mlflow_error(
    registry: ModelRegistry,
    client: MagicMock,
) -> None:
    client.set_model_version_tag.side_effect = MlflowException(
        "Tag failed.",
    )

    with pytest.raises(
        ModelRegistryError,
        match="Could not set tags",
    ):
        registry.set_version_tags(
            version=MODEL_VERSION,
            tags={"model_type": "item_knn"},
        )


def test_get_version_by_alias_wraps_mlflow_error(
    registry: ModelRegistry,
    client: MagicMock,
) -> None:
    client.get_model_version_by_alias.side_effect = MlflowException(
        "Alias not found.",
    )

    with pytest.raises(
        ModelRegistryError,
        match="Could not retrieve alias",
    ):
        registry.get_version_by_alias("staging")


def test_normalize_tags_rejects_empty_key_after_conversion() -> None:
    with pytest.raises(
        ValueError,
        match="keys cannot be empty",
    ):
        _normalize_tags({"   ": "item_knn"})
