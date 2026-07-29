"""Unit tests for early stopping."""

import math

import pytest

from retail_recommender.training.early_stopping import EarlyStopping

DEFAULT_PATIENCE = 2


def test_first_finite_loss_is_an_improvement() -> None:
    early_stopping = EarlyStopping(patience=DEFAULT_PATIENCE)

    improved = early_stopping.update(0.8)

    assert improved
    assert early_stopping.best_loss == pytest.approx(0.8)
    assert early_stopping.bad_epoch_count == 0
    assert not early_stopping.should_stop


def test_improvement_resets_bad_epoch_count() -> None:
    early_stopping = EarlyStopping(patience=DEFAULT_PATIENCE)

    early_stopping.update(0.8)
    early_stopping.update(0.9)
    early_stopping.update(0.7)

    assert early_stopping.bad_epoch_count == 0
    assert not early_stopping.should_stop


def test_early_stopping_triggers_after_patience() -> None:
    early_stopping = EarlyStopping(patience=DEFAULT_PATIENCE)

    early_stopping.update(0.8)
    early_stopping.update(0.9)
    early_stopping.update(1.0)

    assert early_stopping.bad_epoch_count == DEFAULT_PATIENCE
    assert early_stopping.should_stop


def test_minimum_delta_controls_improvement() -> None:
    early_stopping = EarlyStopping(
        patience=DEFAULT_PATIENCE,
        minimum_delta=0.1,
    )

    early_stopping.update(1.0)
    improved = early_stopping.update(0.95)

    assert not improved
    assert early_stopping.best_loss == pytest.approx(1.0)


@pytest.mark.parametrize(
    "patience",
    [0, -1],
)
def test_constructor_rejects_invalid_patience(
    patience: int,
) -> None:
    with pytest.raises(
        ValueError,
        match="patience must be greater than zero",
    ):
        EarlyStopping(patience=patience)


def test_constructor_rejects_negative_minimum_delta() -> None:
    with pytest.raises(
        ValueError,
        match="minimum_delta must be non-negative",
    ):
        EarlyStopping(
            patience=DEFAULT_PATIENCE,
            minimum_delta=-0.1,
        )


@pytest.mark.parametrize(
    "validation_loss",
    [math.inf, -math.inf, math.nan],
)
def test_update_rejects_non_finite_loss(
    validation_loss: float,
) -> None:
    early_stopping = EarlyStopping(patience=DEFAULT_PATIENCE)

    with pytest.raises(
        ValueError,
        match="validation_loss must be finite",
    ):
        early_stopping.update(validation_loss)
