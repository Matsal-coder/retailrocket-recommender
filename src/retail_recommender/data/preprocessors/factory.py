"""Factory for event preprocessing strategies."""

from collections.abc import Mapping, Sequence

from retail_recommender.data.preprocessors.base import EventPreprocessor
from retail_recommender.data.preprocessors.implicit_feedback import (
    ImplicitFeedbackPreprocessor,
)

IMPLICIT_FEEDBACK_STRATEGY = "implicit_feedback"


class EventPreprocessorFactory:
    """Create event preprocessors from configuration values."""

    @staticmethod
    def create(
        strategy: str,
        event_weights: Mapping[str, float],
        allowed_event_types: Sequence[str] | None = None,
    ) -> EventPreprocessor:
        """Create a configured preprocessing strategy.

        Args:
            strategy: Name of the preprocessing strategy.
            event_weights: Weight assigned to each supported event.
            allowed_event_types: Event types allowed in the output.

        Returns:
            A configured event preprocessing strategy.

        Raises:
            ValueError: If the requested strategy is not supported.
        """
        normalized_strategy = strategy.strip().lower()

        if normalized_strategy == IMPLICIT_FEEDBACK_STRATEGY:
            return ImplicitFeedbackPreprocessor(
                event_weights=event_weights,
                allowed_event_types=allowed_event_types,
            )

        msg = f"Unsupported preprocessing strategy: {strategy}."
        raise ValueError(msg)
