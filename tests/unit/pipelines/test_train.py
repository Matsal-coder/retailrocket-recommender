"""Unit tests for the training pipeline helpers."""

import pandas as pd
import pytest
import torch

from retail_recommender.pipelines.train import (
    _get_required_float,
    _get_required_int,
    _get_required_int_sequence,
    _get_required_string,
    _resolve_device,
    _resolve_entity_counts,
    _validate_interactions,
)

EXPECTED_USER_COUNT = 4
EXPECTED_ITEM_COUNT = 6
DEFAULT_EPOCHS = 3


def test_resolve_entity_counts_uses_largest_encoded_ids() -> None:
    train_interactions = pd.DataFrame(
        {
            "user_idx": [0, 1, 2],
            "item_idx": [0, 2, 4],
            "target": [1, 0, 1],
        }
    )
    validation_interactions = pd.DataFrame(
        {
            "user_idx": [3],
            "item_idx": [5],
            "target": [1],
        }
    )

    num_users, num_items = _resolve_entity_counts(
        train_interactions,
        validation_interactions,
    )

    assert num_users == EXPECTED_USER_COUNT
    assert num_items == EXPECTED_ITEM_COUNT


def test_validate_interactions_rejects_missing_columns() -> None:
    interactions = pd.DataFrame(
        {
            "user_idx": [0],
            "item_idx": [1],
        }
    )

    with pytest.raises(
        ValueError,
        match="training data is missing columns: target",
    ):
        _validate_interactions(
            interactions,
            split_name="training",
        )


def test_validate_interactions_rejects_empty_split() -> None:
    interactions = pd.DataFrame(
        columns=[
            "user_idx",
            "item_idx",
            "target",
        ]
    )

    with pytest.raises(
        ValueError,
        match="validation data must not be empty",
    ):
        _validate_interactions(
            interactions,
            split_name="validation",
        )


@pytest.mark.parametrize(
    ("device_name", "expected_type"),
    [
        ("cpu", "cpu"),
        (" CPU ", "cpu"),
    ],
)
def test_resolve_device_returns_cpu(
    device_name: str,
    expected_type: str,
) -> None:
    device = _resolve_device(device_name)

    assert device.type == expected_type


def test_resolve_device_auto_returns_valid_device() -> None:
    device = _resolve_device("auto")

    assert device.type in {"cpu", "cuda"}


def test_resolve_device_rejects_unknown_name() -> None:
    with pytest.raises(
        ValueError,
        match="device must be one of",
    ):
        _resolve_device("tpu")


def test_resolve_device_rejects_unavailable_cuda(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        torch.cuda,
        "is_available",
        lambda: False,
    )

    with pytest.raises(
        RuntimeError,
        match="CUDA was requested but is not available",
    ):
        _resolve_device("cuda")


def test_get_required_configuration_values() -> None:
    values = {
        "path": "model.pt",
        "epochs": DEFAULT_EPOCHS,
        "dropout": 0.2,
        "hidden_layers": [8, 4],
    }

    assert _get_required_string(values, "path") == "model.pt"
    assert _get_required_int(values, "epochs") == DEFAULT_EPOCHS
    assert _get_required_float(values, "dropout") == pytest.approx(0.2)
    assert _get_required_int_sequence(
        values,
        "hidden_layers",
    ) == (8, 4)
