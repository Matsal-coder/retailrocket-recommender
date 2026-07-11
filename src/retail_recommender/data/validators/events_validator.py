"""Validation utilities for RetailRocket events data."""

from dataclasses import dataclass, field
from typing import Any

import pandas as pd

REQUIRED_COLUMNS: tuple[str, ...] = (
    "timestamp",
    "visitorid",
    "event",
    "itemid",
    "transactionid",
)

ALLOWED_EVENTS: frozenset[str] = frozenset(
    {
        "view",
        "addtocart",
        "transaction",
    }
)


@dataclass(frozen=True)
class EventsValidationResult:
    """Structured result for RetailRocket events validation."""

    is_valid: bool
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    summary: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert validation result to a JSON-serializable dictionary."""
        return {
            "is_valid": self.is_valid,
            "errors": self.errors,
            "warnings": self.warnings,
            "summary": self.summary,
        }


@dataclass(frozen=True)
class EventsValidationConfig:
    """Configuration for RetailRocket events validation."""

    required_columns: tuple[str, ...] = REQUIRED_COLUMNS
    allowed_events: frozenset[str] = ALLOWED_EVENTS
    minimum_interactions: int = 10_000
    minimum_users: int = 100
    minimum_items: int = 100


def validate_events(
    events: pd.DataFrame,
    config: EventsValidationConfig | None = None,
) -> EventsValidationResult:
    """Validate the structure and minimum quality of RetailRocket events.

    Parameters
    ----------
    events:
        Raw RetailRocket events dataframe.
    config:
        Optional validation configuration.

    Returns
    -------
    EventsValidationResult
        Structured validation result with errors, warnings and summary.
    """
    validation_config = config or EventsValidationConfig()

    errors: list[str] = []
    warnings: list[str] = []

    _validate_required_columns(events, validation_config.required_columns, errors)

    if errors:
        return EventsValidationResult(
            is_valid=False,
            errors=errors,
            warnings=warnings,
            summary=_build_basic_summary(events),
        )

    _validate_required_columns_not_fully_empty(
        events=events,
        required_columns=("timestamp", "visitorid", "event", "itemid"),
        errors=errors,
    )
    _validate_allowed_events(
        events=events,
        allowed_events=validation_config.allowed_events,
        errors=errors,
    )
    _validate_timestamp_conversion(events=events, errors=errors)
    _validate_minimum_interactions(
        events=events,
        minimum_interactions=validation_config.minimum_interactions,
        errors=errors,
    )
    _validate_minimum_unique_values(
        events=events,
        column="visitorid",
        minimum=validation_config.minimum_users,
        label="users",
        errors=errors,
    )
    _validate_minimum_unique_values(
        events=events,
        column="itemid",
        minimum=validation_config.minimum_items,
        label="items",
        errors=errors,
    )

    summary = _build_validation_summary(events)

    return EventsValidationResult(
        is_valid=len(errors) == 0,
        errors=errors,
        warnings=warnings,
        summary=summary,
    )


def _validate_required_columns(
    events: pd.DataFrame,
    required_columns: tuple[str, ...],
    errors: list[str],
) -> None:
    missing_columns = [
        column for column in required_columns if column not in events.columns
    ]

    if missing_columns:
        errors.append("Missing required columns: " + ", ".join(sorted(missing_columns)))


def _validate_required_columns_not_fully_empty(
    events: pd.DataFrame,
    required_columns: tuple[str, ...],
    errors: list[str],
) -> None:
    empty_columns = [
        column for column in required_columns if events[column].isna().all()
    ]

    if empty_columns:
        errors.append(
            "Required columns are fully empty: " + ", ".join(sorted(empty_columns))
        )


def _validate_allowed_events(
    events: pd.DataFrame,
    allowed_events: frozenset[str],
    errors: list[str],
) -> None:
    event_values = set(events["event"].dropna().unique())
    unexpected_events = event_values - allowed_events

    if unexpected_events:
        errors.append(
            "Unexpected event values found: "
            + ", ".join(sorted(str(event) for event in unexpected_events))
        )


def _validate_timestamp_conversion(
    events: pd.DataFrame,
    errors: list[str],
) -> None:
    converted_timestamp = pd.to_datetime(
        events["timestamp"],
        unit="ms",
        errors="coerce",
    )

    if converted_timestamp.isna().all():
        errors.append("Timestamp column cannot be converted to datetime.")


def _validate_minimum_interactions(
    events: pd.DataFrame,
    minimum_interactions: int,
    errors: list[str],
) -> None:
    total_interactions = len(events)

    if total_interactions < minimum_interactions:
        errors.append(
            f"Dataset has {total_interactions} interactions, "
            f"but at least {minimum_interactions} are required."
        )


def _validate_minimum_unique_values(
    events: pd.DataFrame,
    column: str,
    minimum: int,
    label: str,
    errors: list[str],
) -> None:
    unique_values = events[column].nunique(dropna=True)

    if unique_values < minimum:
        errors.append(
            f"Dataset has {unique_values} unique {label}, "
            f"but at least {minimum} are required."
        )


def _build_basic_summary(events: pd.DataFrame) -> dict[str, Any]:
    return {
        "rows": int(len(events)),
        "columns": list(events.columns),
    }


def _build_validation_summary(events: pd.DataFrame) -> dict[str, Any]:
    return {
        "rows": int(len(events)),
        "columns": list(events.columns),
        "unique_users": int(events["visitorid"].nunique(dropna=True)),
        "unique_items": int(events["itemid"].nunique(dropna=True)),
        "event_counts": {
            str(event): int(count)
            for event, count in events["event"].value_counts(dropna=False).items()
        },
        "min_timestamp": _safe_timestamp_min(events),
        "max_timestamp": _safe_timestamp_max(events),
    }


def _safe_timestamp_min(events: pd.DataFrame) -> str | None:
    converted_timestamp = pd.to_datetime(
        events["timestamp"],
        unit="ms",
        errors="coerce",
    )

    if converted_timestamp.dropna().empty:
        return None

    return converted_timestamp.min().isoformat()


def _safe_timestamp_max(events: pd.DataFrame) -> str | None:
    converted_timestamp = pd.to_datetime(
        events["timestamp"],
        unit="ms",
        errors="coerce",
    )

    if converted_timestamp.dropna().empty:
        return None

    return converted_timestamp.max().isoformat()
