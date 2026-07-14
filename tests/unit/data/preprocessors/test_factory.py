"""Tests for the event preprocessor factory."""

import pytest

from retail_recommender.data.preprocessors.factory import (
    EventPreprocessorFactory,
)
from retail_recommender.data.preprocessors.implicit_feedback import (
    ImplicitFeedbackPreprocessor,
)

EVENT_WEIGHTS = {
    "view": 1.0,
    "addtocart": 3.0,
    "transaction": 5.0,
}


def test_factory_creates_implicit_feedback_preprocessor() -> None:
    """It should create the configured implicit-feedback strategy."""
    preprocessor = EventPreprocessorFactory.create(
        strategy="implicit_feedback",
        event_weights=EVENT_WEIGHTS,
        allowed_event_types=list(EVENT_WEIGHTS),
    )

    assert isinstance(preprocessor, ImplicitFeedbackPreprocessor)


def test_factory_normalizes_strategy_name() -> None:
    """It should normalize whitespace and capitalization."""
    preprocessor = EventPreprocessorFactory.create(
        strategy=" Implicit_Feedback ",
        event_weights=EVENT_WEIGHTS,
    )

    assert isinstance(preprocessor, ImplicitFeedbackPreprocessor)


def test_factory_raises_error_for_unsupported_strategy() -> None:
    """It should reject unknown preprocessing strategies."""
    with pytest.raises(ValueError, match="Unsupported preprocessing strategy"):
        EventPreprocessorFactory.create(
            strategy="unknown_strategy",
            event_weights=EVENT_WEIGHTS,
        )
