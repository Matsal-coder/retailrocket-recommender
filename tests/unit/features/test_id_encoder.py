"""Tests for user and item ID encoding."""

from pathlib import Path

import pandas as pd
import pytest

from retail_recommender.features.id_encoder import (
    UNKNOWN_INDEX,
    IdEncoder,
    fit_interaction_encoders,
    transform_interaction_ids,
)

USER_A = 10
USER_B = 20
USER_C = 30
USER_UNKNOWN = 99

ITEM_X = 100
ITEM_Y = 200
ITEM_UNKNOWN = 999

EXPECTED_ENCODER_SIZE = 3
EXPECTED_FIRST_INDEX = 0
EXPECTED_SECOND_INDEX = 1
EXPECTED_ITEM_ENCODER_SIZE = 2
EXPECTED_ENCODER_SIZE_TRANSFORM_COMBINES_OPERATIONS = 2


def make_train_interactions() -> pd.DataFrame:
    """Create training interactions for encoder tests."""
    return pd.DataFrame(
        {
            "user_id": [
                USER_A,
                USER_B,
                USER_A,
                USER_C,
            ],
            "item_id": [
                ITEM_X,
                ITEM_Y,
                ITEM_Y,
                ITEM_X,
            ],
            "target": [1, 1, 1, 1],
        }
    )


def test_id_encoder_fits_unique_values() -> None:
    """It should create one continuous index per unique identifier."""
    encoder = IdEncoder().fit(
        [USER_A, USER_B, USER_A, USER_C],
    )

    assert encoder.is_fitted
    assert encoder.size == EXPECTED_ENCODER_SIZE
    assert encoder.id_to_index[USER_A] == EXPECTED_FIRST_INDEX
    assert encoder.id_to_index[USER_B] == EXPECTED_SECOND_INDEX


def test_id_encoder_preserves_first_occurrence_order() -> None:
    """It should assign indices according to first occurrence."""
    encoder = IdEncoder().fit(
        [USER_B, USER_A, USER_C],
    )

    assert encoder.id_to_index == {
        USER_B: 0,
        USER_A: 1,
        USER_C: 2,
    }


def test_id_encoder_transforms_known_values() -> None:
    """It should transform known IDs into integer indices."""
    encoder = IdEncoder().fit(
        [USER_A, USER_B, USER_C],
    )

    result = encoder.transform(
        [USER_C, USER_A, USER_B],
    )

    assert result.tolist() == [2, 0, 1]
    assert str(result.dtype) == "int64"


def test_id_encoder_assigns_unknown_index() -> None:
    """It should assign the configured index to unknown IDs."""
    encoder = IdEncoder().fit([USER_A, USER_B])

    result = encoder.transform(
        [USER_A, USER_UNKNOWN, None],
    )

    assert result.tolist() == [
        EXPECTED_FIRST_INDEX,
        UNKNOWN_INDEX,
        UNKNOWN_INDEX,
    ]


def test_id_encoder_inverse_transforms_indices() -> None:
    """It should restore original IDs from known indices."""
    encoder = IdEncoder().fit(
        [USER_A, USER_B, USER_C],
    )

    result = encoder.inverse_transform([2, 0, 1])

    assert result.tolist() == [
        USER_C,
        USER_A,
        USER_B,
    ]


def test_id_encoder_fit_transform_combines_operations() -> None:
    """It should fit and transform the same values."""
    encoder = IdEncoder()

    result = encoder.fit_transform(
        [USER_A, USER_B, USER_A],
    )

    assert result.tolist() == [0, 1, 0]
    assert encoder.size == EXPECTED_ENCODER_SIZE_TRANSFORM_COMBINES_OPERATIONS


def test_id_encoder_rejects_empty_values() -> None:
    """It should reject input without valid IDs."""
    with pytest.raises(ValueError, match="without valid identifiers"):
        IdEncoder().fit([None, None])


def test_id_encoder_rejects_transform_before_fit() -> None:
    """It should require fitting before transformation."""
    with pytest.raises(RuntimeError, match="must be fitted"):
        IdEncoder().transform([USER_A])


def test_id_encoder_rejects_inverse_transform_before_fit() -> None:
    """It should require fitting before inverse transformation."""
    with pytest.raises(RuntimeError, match="must be fitted"):
        IdEncoder().inverse_transform([0])


def test_fit_interaction_encoders_uses_training_ids() -> None:
    """It should fit separate user and item mappings."""
    train = make_train_interactions()

    user_encoder, item_encoder = fit_interaction_encoders(train)

    assert user_encoder.size == EXPECTED_ENCODER_SIZE
    assert item_encoder.size == EXPECTED_ITEM_ENCODER_SIZE


def test_transform_interaction_ids_adds_encoded_columns() -> None:
    """It should add user_idx and item_idx to interactions."""
    train = make_train_interactions()
    user_encoder, item_encoder = fit_interaction_encoders(train)

    result = transform_interaction_ids(
        interactions=train,
        user_encoder=user_encoder,
        item_encoder=item_encoder,
    )

    assert "user_idx" in result.columns
    assert "item_idx" in result.columns
    assert result["user_idx"].ge(0).all()
    assert result["item_idx"].ge(0).all()


def test_transform_interaction_ids_drops_unknown_rows() -> None:
    """It should remove rows containing unknown entities."""
    train = make_train_interactions()
    user_encoder, item_encoder = fit_interaction_encoders(train)

    future = pd.DataFrame(
        {
            "user_id": [USER_A, USER_UNKNOWN],
            "item_id": [ITEM_X, ITEM_UNKNOWN],
            "target": [1, 1],
        }
    )

    result = transform_interaction_ids(
        interactions=future,
        user_encoder=user_encoder,
        item_encoder=item_encoder,
        drop_unknowns=True,
    )

    assert len(result) == 1
    assert result.loc[0, "user_id"] == USER_A
    assert result.loc[0, "item_id"] == ITEM_X


def test_transform_interaction_ids_can_keep_unknown_rows() -> None:
    """It should keep unknown rows when explicitly requested."""
    train = make_train_interactions()
    user_encoder, item_encoder = fit_interaction_encoders(train)

    future = pd.DataFrame(
        {
            "user_id": [USER_UNKNOWN],
            "item_id": [ITEM_UNKNOWN],
            "target": [1],
        }
    )

    result = transform_interaction_ids(
        interactions=future,
        user_encoder=user_encoder,
        item_encoder=item_encoder,
        drop_unknowns=False,
    )

    assert result.loc[0, "user_idx"] == UNKNOWN_INDEX
    assert result.loc[0, "item_idx"] == UNKNOWN_INDEX


def test_id_encoder_persists_and_loads_mapping(
    tmp_path: Path,
) -> None:
    """It should preserve mappings across persistence."""
    encoder_path = tmp_path / "user_encoder.pkl"
    original = IdEncoder().fit(
        [USER_A, USER_B, USER_C],
    )

    original.save(encoder_path)
    loaded = IdEncoder.load(encoder_path)

    assert encoder_path.exists()
    assert loaded.is_fitted
    assert loaded.id_to_index == original.id_to_index
    assert loaded.transform([USER_C]).tolist() == [2]


def test_id_encoder_load_rejects_missing_file(
    tmp_path: Path,
) -> None:
    """It should reject an absent encoder file."""
    missing_path = tmp_path / "missing.pkl"

    with pytest.raises(FileNotFoundError, match="not found"):
        IdEncoder.load(missing_path)


def test_encoding_helpers_reject_missing_id_columns() -> None:
    """They should require user_id and item_id."""
    invalid = pd.DataFrame({"target": [1]})

    with pytest.raises(ValueError, match="user_id"):
        fit_interaction_encoders(invalid)
