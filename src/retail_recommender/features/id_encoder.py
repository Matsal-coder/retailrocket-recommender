"""Encode original entity IDs as continuous integer indices."""

from __future__ import annotations

import pickle
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import pandas as pd

UNKNOWN_INDEX = -1


class IdEncoder:
    """Encode entity identifiers into continuous integer indices."""

    def __init__(self) -> None:
        """Initialize an unfitted ID encoder."""
        self._id_to_index: dict[Any, int] = {}
        self._index_to_id: dict[int, Any] = {}
        self._is_fitted = False

    @property
    def is_fitted(self) -> bool:
        """Return whether the encoder has already been fitted."""
        return self._is_fitted

    @property
    def size(self) -> int:
        """Return the number of known identifiers."""
        return len(self._id_to_index)

    @property
    def id_to_index(self) -> dict[Any, int]:
        """Return a copy of the ID-to-index mapping."""
        return self._id_to_index.copy()

    def fit(self, values: Iterable[Any]) -> IdEncoder:
        """Fit the encoder using unique non-null identifiers.

        Args:
            values: Entity identifiers used to build the mapping.

        Returns:
            The fitted encoder.

        Raises:
            ValueError: If no valid identifiers are provided.
        """
        normalized_values = pd.Series(values).dropna().drop_duplicates()

        if normalized_values.empty:
            msg = "Cannot fit ID encoder without valid identifiers."
            raise ValueError(msg)

        unique_values = normalized_values.tolist()

        self._id_to_index = {
            entity_id: index for index, entity_id in enumerate(unique_values)
        }
        self._index_to_id = {
            index: entity_id for entity_id, index in self._id_to_index.items()
        }
        self._is_fitted = True

        return self

    def transform(
        self,
        values: Iterable[Any],
        unknown_index: int = UNKNOWN_INDEX,
    ) -> pd.Series:
        """Transform original identifiers into integer indices.

        Args:
            values: Original entity identifiers.
            unknown_index: Index assigned to unknown or missing identifiers.

        Returns:
            Integer indices aligned with the input order.

        Raises:
            RuntimeError: If the encoder has not been fitted.
        """
        self._validate_is_fitted()

        source = values.copy() if isinstance(values, pd.Series) else pd.Series(values)
        encoded = source.map(self._id_to_index)
        encoded = encoded.fillna(unknown_index).astype("int64")

        return encoded

    def fit_transform(self, values: Iterable[Any]) -> pd.Series:
        """Fit the encoder and transform the provided identifiers.

        Args:
            values: Entity identifiers.

        Returns:
            Encoded integer indices.
        """
        self.fit(values)
        return self.transform(values)

    def inverse_transform(
        self,
        indices: Iterable[int],
    ) -> pd.Series:
        """Convert integer indices back to original identifiers.

        Args:
            indices: Encoded integer indices.

        Returns:
            Original identifiers. Unknown indices become missing values.

        Raises:
            RuntimeError: If the encoder has not been fitted.
        """
        self._validate_is_fitted()

        return pd.Series(indices).map(self._index_to_id)

    def save(self, path: str | Path) -> None:
        """Persist the fitted encoder as a pickle file.

        Args:
            path: Destination file path.

        Raises:
            RuntimeError: If the encoder has not been fitted.
        """
        self._validate_is_fitted()

        output_path = Path(path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with output_path.open("wb") as file:
            pickle.dump(self, file)

    @classmethod
    def load(cls, path: str | Path) -> IdEncoder:
        """Load an encoder from a pickle file.

        Args:
            path: Persisted encoder path.

        Returns:
            Loaded encoder.

        Raises:
            FileNotFoundError: If the encoder file does not exist.
            TypeError: If the loaded object is not an ID encoder.
        """
        input_path = Path(path)

        if not input_path.exists():
            msg = f"Encoder file not found: {input_path}"
            raise FileNotFoundError(msg)

        with input_path.open("rb") as file:
            encoder = pickle.load(file)

        if not isinstance(encoder, cls):
            msg = f"Invalid encoder object stored at: {input_path}"
            raise TypeError(msg)

        return encoder

    def _validate_is_fitted(self) -> None:
        """Raise an error if the encoder is not fitted."""
        if not self._is_fitted:
            msg = "ID encoder must be fitted before this operation."
            raise RuntimeError(msg)


def fit_interaction_encoders(
    train: pd.DataFrame,
) -> tuple[IdEncoder, IdEncoder]:
    """Fit user and item encoders using training interactions only.

    Args:
        train: Training interaction dataset.

    Returns:
        Fitted user and item encoders.

    Raises:
        ValueError: If required ID columns are missing.
    """
    _validate_id_columns(train)

    user_encoder = IdEncoder().fit(train["user_id"])
    item_encoder = IdEncoder().fit(train["item_id"])

    return user_encoder, item_encoder


def transform_interaction_ids(
    interactions: pd.DataFrame,
    user_encoder: IdEncoder,
    item_encoder: IdEncoder,
    drop_unknowns: bool = True,
) -> pd.DataFrame:
    """Add encoded user and item indices to interactions.

    Args:
        interactions: Interaction dataset.
        user_encoder: Fitted user ID encoder.
        item_encoder: Fitted item ID encoder.
        drop_unknowns: Whether rows with unknown entities must be removed.

    Returns:
        Interactions with user_idx and item_idx columns.

    Raises:
        ValueError: If required ID columns are missing.
    """
    _validate_id_columns(interactions)

    transformed = interactions.copy()

    transformed["user_idx"] = user_encoder.transform(
        transformed["user_id"],
    )
    transformed["item_idx"] = item_encoder.transform(
        transformed["item_id"],
    )

    if drop_unknowns:
        known_mask = transformed["user_idx"].ne(UNKNOWN_INDEX) & transformed[
            "item_idx"
        ].ne(UNKNOWN_INDEX)
        transformed = transformed.loc[known_mask].copy()

    return transformed.reset_index(drop=True)


def _validate_id_columns(interactions: pd.DataFrame) -> None:
    """Validate user and item columns before encoding."""
    required_columns = {"user_id", "item_id"}
    missing_columns = required_columns - set(interactions.columns)

    if missing_columns:
        missing = ", ".join(sorted(missing_columns))
        msg = f"Missing required ID columns: {missing}."
        raise ValueError(msg)
