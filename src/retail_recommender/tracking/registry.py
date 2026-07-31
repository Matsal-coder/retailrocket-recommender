"""MLflow Model Registry integration."""

from __future__ import annotations

import logging
from collections.abc import Mapping
from dataclasses import dataclass

from mlflow import MlflowClient
from mlflow.entities.model_registry import ModelVersion
from mlflow.exceptions import MlflowException

from retail_recommender.config.settings import Settings, get_settings

LOGGER = logging.getLogger(__name__)

RESOURCE_DOES_NOT_EXIST_ERROR_CODE = "RESOURCE_DOES_NOT_EXIST"


class ModelRegistryError(RuntimeError):
    """Raised when an MLflow Model Registry operation fails."""


@dataclass(frozen=True)
class RegisteredVersion:
    """Metadata returned after creating a registered model version."""

    model_name: str
    version: str
    source: str
    run_id: str | None


class ModelRegistry:
    """Manage model versions and aliases in the MLflow Model Registry."""

    def __init__(
        self,
        settings: Settings | None = None,
        client: MlflowClient | None = None,
    ) -> None:
        """Initialize the registry using centralized application settings."""

        self._settings = settings or get_settings()
        self._client = client or MlflowClient(
            tracking_uri=self._settings.mlflow_tracking_uri,
            registry_uri=self._settings.mlflow_tracking_uri,
        )

    @property
    def model_name(self) -> str:
        """Return the configured registered model name."""

        return self._settings.mlflow_registered_model_name

    def ensure_registered_model(
        self,
        description: str | None = None,
    ) -> None:
        """Create the registered model when it does not already exist."""

        try:
            self._client.get_registered_model(self.model_name)
        except MlflowException as exc:
            if exc.error_code != RESOURCE_DOES_NOT_EXIST_ERROR_CODE:
                message = f"Could not access registered model " f"'{self.model_name}'."
                raise ModelRegistryError(message) from exc

            try:
                self._client.create_registered_model(
                    name=self.model_name,
                    description=description,
                )
            except MlflowException as create_exc:
                message = f"Could not create registered model " f"'{self.model_name}'."
                raise ModelRegistryError(message) from create_exc

            LOGGER.info(
                "Created registered model '%s'.",
                self.model_name,
            )
            return

        if description is not None:
            try:
                self._client.update_registered_model(
                    name=self.model_name,
                    description=description,
                )
            except MlflowException as exc:
                message = f"Could not update registered model " f"'{self.model_name}'."
                raise ModelRegistryError(message) from exc

        LOGGER.info(
            "Registered model '%s' already exists.",
            self.model_name,
        )

    def create_version(
        self,
        *,
        model_uri: str,
        run_id: str | None = None,
        description: str | None = None,
        tags: Mapping[str, str] | None = None,
    ) -> RegisteredVersion:
        """Create a model version from an MLflow model URI."""

        if not model_uri.strip():
            raise ValueError("Model URI cannot be empty.")

        normalized_tags = _normalize_tags(tags)

        try:
            version = self._client.create_model_version(
                name=self.model_name,
                source=model_uri,
                run_id=run_id,
                description=description,
                tags=normalized_tags,
            )
        except MlflowException as exc:
            message = (
                f"Could not create a version of registered model "
                f"'{self.model_name}' from URI '{model_uri}'."
            )
            raise ModelRegistryError(message) from exc

        LOGGER.info(
            "Created version %s of registered model '%s'.",
            version.version,
            self.model_name,
        )

        return RegisteredVersion(
            model_name=self.model_name,
            version=str(version.version),
            source=model_uri,
            run_id=run_id,
        )

    def set_version_tags(
        self,
        *,
        version: str,
        tags: Mapping[str, str],
    ) -> None:
        """Set metadata tags on a registered model version."""

        normalized_version = _normalize_version(version)
        normalized_tags = _normalize_tags(tags)

        try:
            for key, value in normalized_tags.items():
                self._client.set_model_version_tag(
                    name=self.model_name,
                    version=normalized_version,
                    key=key,
                    value=value,
                )
        except MlflowException as exc:
            message = (
                f"Could not set tags on version {normalized_version} "
                f"of registered model '{self.model_name}'."
            )
            raise ModelRegistryError(message) from exc

    def set_alias(
        self,
        *,
        version: str,
        alias: str,
    ) -> None:
        """Assign an alias to a registered model version."""

        normalized_version = _normalize_version(version)
        normalized_alias = _normalize_alias(alias)

        try:
            self._client.set_registered_model_alias(
                name=self.model_name,
                alias=normalized_alias,
                version=normalized_version,
            )
        except MlflowException as exc:
            message = (
                f"Could not assign alias '{normalized_alias}' "
                f"to version {normalized_version} of registered model "
                f"'{self.model_name}'."
            )
            raise ModelRegistryError(message) from exc

        LOGGER.info(
            "Assigned alias '%s' to version %s of model '%s'.",
            normalized_alias,
            normalized_version,
            self.model_name,
        )

    def promote_to_staging(self, version: str) -> None:
        """Assign the configured staging alias to a model version."""

        self.set_alias(
            version=version,
            alias=self._settings.mlflow_staging_alias,
        )

    def promote_to_production(self, version: str) -> None:
        """Assign the configured production alias to a model version."""

        self.set_alias(
            version=version,
            alias=self._settings.mlflow_production_alias,
        )

    def get_version_by_alias(
        self,
        alias: str,
    ) -> ModelVersion:
        """Return the model version associated with an alias."""

        normalized_alias = _normalize_alias(alias)

        try:
            return self._client.get_model_version_by_alias(
                name=self.model_name,
                alias=normalized_alias,
            )
        except MlflowException as exc:
            message = (
                f"Could not retrieve alias '{normalized_alias}' "
                f"from registered model '{self.model_name}'."
            )
            raise ModelRegistryError(message) from exc


def _normalize_version(version: object) -> str:
    """Validate and normalize a model version identifier."""

    normalized = str(version).strip()

    if not normalized:
        raise ValueError("Model version cannot be empty.")

    return normalized


def _normalize_alias(alias: str) -> str:
    """Validate and normalize a model alias."""

    normalized = alias.strip()

    if not normalized:
        raise ValueError("Model alias cannot be empty.")

    return normalized


def _normalize_tags(
    tags: Mapping[object, object] | None,
) -> dict[str, str]:
    """Validate and normalize model version tags."""

    if tags is None:
        return {}

    normalized: dict[str, str] = {}

    for key, value in tags.items():
        normalized_key = str(key).strip()
        normalized_value = str(value).strip()

        if not normalized_key:
            raise ValueError("Model version tag keys cannot be empty.")

        if not normalized_value:
            message = (
                f"Model version tag '{normalized_key}' " "cannot have an empty value."
            )
            raise ValueError(message)

        normalized[normalized_key] = normalized_value

    return normalized
