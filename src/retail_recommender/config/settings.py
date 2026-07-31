"""Application settings loaded from environment variables."""

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

AppEnvironment = Literal["local", "dev", "staging", "prod"]
DEFAULT_MLFLOW_TRACKING_URI = "sqlite:///mlflow.db"
DEFAULT_MLFLOW_EXPERIMENT_NAME = "retailrocket-recommender"


class Settings(BaseSettings):
    """Application settings.

    Values are loaded from environment variables and optionally from a local
    `.env` file. The `.env` file is useful during local development, while
    production-like environments should prefer real environment variables.
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        env_ignore_empty=True,
        extra="ignore",
    )

    app_name: str = Field(default="retailrocket-recommender", alias="APP_NAME")
    app_env: AppEnvironment = Field(default="local", alias="APP_ENV")
    log_level: str = Field(default="INFO", alias="LOG_LEVEL")

    data_dir: Path = Field(default=Path("data"), alias="DATA_DIR")
    raw_data_dir: Path = Field(default=Path("data/raw"), alias="RAW_DATA_DIR")
    interim_data_dir: Path = Field(
        default=Path("data/interim"),
        alias="INTERIM_DATA_DIR",
    )
    processed_data_dir: Path = Field(
        default=Path("data/processed"),
        alias="PROCESSED_DATA_DIR",
    )
    artifacts_dir: Path = Field(default=Path("artifacts"), alias="ARTIFACTS_DIR")

    mlflow_tracking_uri: str | None = Field(
        default=DEFAULT_MLFLOW_TRACKING_URI,
        alias="MLFLOW_TRACKING_URI",
    )
    mlflow_experiment_name: str = Field(
        default=DEFAULT_MLFLOW_EXPERIMENT_NAME,
        alias="MLFLOW_EXPERIMENT_NAME",
    )


@lru_cache
def get_settings() -> Settings:
    """Return cached application settings.

    Caching avoids re-reading environment variables repeatedly across the
    application.
    """
    return Settings()
