"""Generate reproducible negative user-item interactions."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

REQUIRED_COLUMNS = {
    "user_idx",
    "item_idx",
    "target",
}

POSITIVE_TARGET = 1
NEGATIVE_TARGET = 0


@dataclass(frozen=True)
class NegativeSamplingConfig:
    """Configure negative sampling for implicit-feedback training."""

    negative_samples_per_positive: int
    random_seed: int

    def __post_init__(self) -> None:
        """Validate negative sampling parameters."""
        if self.negative_samples_per_positive < 1:
            msg = "negative_samples_per_positive must be at least 1."
            raise ValueError(msg)


def generate_negative_samples(
    positive_interactions: pd.DataFrame,
    item_count: int,
    config: NegativeSamplingConfig,
) -> pd.DataFrame:
    """Generate negative user-item pairs for training.

    Args:
        positive_interactions: Encoded positive training interactions.
        item_count: Total number of known item indices.
        config: Negative sampling parameters.

    Returns:
        Negative user-item pairs with target equal to zero.

    Raises:
        ValueError: If the input is invalid or sampling is impossible.
    """
    _validate_positive_interactions(
        positive_interactions=positive_interactions,
        item_count=item_count,
    )

    rng = np.random.default_rng(config.random_seed)

    positive_pairs = set(
        zip(
            positive_interactions["user_idx"].astype(int),
            positive_interactions["item_idx"].astype(int),
            strict=True,
        )
    )

    user_positive_items = _build_user_positive_items(positive_interactions)

    sampled_pairs: list[tuple[int, int]] = []

    for user_idx, user_rows in positive_interactions.groupby(
        "user_idx",
        sort=False,
    ):
        positive_count = len(user_rows)
        requested_negative_count = positive_count * config.negative_samples_per_positive

        user_idx_int = int(user_idx)
        forbidden_items = user_positive_items[user_idx_int]
        available_items = item_count - len(forbidden_items)

        if available_items <= 0:
            msg = (
                f"Cannot generate negatives for user {user_idx_int}: "
                "the user interacted with every known item."
            )
            raise ValueError(msg)

        sampled_items = _sample_user_items(
            rng=rng,
            forbidden_items=forbidden_items,
            item_count=item_count,
            sample_count=requested_negative_count,
        )

        sampled_pairs.extend((user_idx_int, item_idx) for item_idx in sampled_items)

    negatives = pd.DataFrame(
        sampled_pairs,
        columns=["user_idx", "item_idx"],
    )
    negatives["target"] = NEGATIVE_TARGET

    _validate_no_positive_overlap(
        negatives=negatives,
        positive_pairs=positive_pairs,
    )

    return negatives.astype(
        {
            "user_idx": "int64",
            "item_idx": "int64",
            "target": "int8",
        }
    )


def combine_positive_and_negative_interactions(
    positive_interactions: pd.DataFrame,
    negative_interactions: pd.DataFrame,
    random_seed: int,
) -> pd.DataFrame:
    """Combine and shuffle positive and negative training rows.

    Args:
        positive_interactions: Positive encoded interactions.
        negative_interactions: Generated negative interactions.
        random_seed: Seed used for deterministic row shuffling.

    Returns:
        Combined training interactions.
    """
    _validate_required_columns(positive_interactions)
    _validate_required_columns(negative_interactions)

    training_columns = ["user_idx", "item_idx", "target"]

    combined = pd.concat(
        [
            positive_interactions.loc[:, training_columns],
            negative_interactions.loc[:, training_columns],
        ],
        ignore_index=True,
    )

    return combined.sample(
        frac=1.0,
        random_state=random_seed,
    ).reset_index(drop=True)


def _build_user_positive_items(
    positive_interactions: pd.DataFrame,
) -> dict[int, set[int]]:
    """Build a positive-item lookup for every user."""
    return {
        int(user_idx): set(group["item_idx"].astype(int))
        for user_idx, group in positive_interactions.groupby(
            "user_idx",
            sort=False,
        )
    }


def _sample_user_items(
    rng: np.random.Generator,
    forbidden_items: set[int],
    item_count: int,
    sample_count: int,
) -> list[int]:
    """Sample unseen items for one user.

    Sampling allows repeated negative items when the requested number exceeds
    the number of available unseen items.

    Args:
        rng: NumPy random generator.
        user_idx: Encoded user index.
        forbidden_items: Positive items already observed by the user.
        item_count: Total number of known item indices.
        sample_count: Number of negatives requested for the user.

    Returns:
        Sampled unseen item indices.
    """
    available_items = np.fromiter(
        (item_idx for item_idx in range(item_count) if item_idx not in forbidden_items),
        dtype=np.int64,
    )

    replace = sample_count > len(available_items)

    sampled = rng.choice(
        available_items,
        size=sample_count,
        replace=replace,
    )

    return sampled.astype(int).tolist()


def _validate_positive_interactions(
    positive_interactions: pd.DataFrame,
    item_count: int,
) -> None:
    """Validate encoded positive interactions before sampling."""
    _validate_required_columns(positive_interactions)

    if positive_interactions.empty:
        msg = "Cannot generate negatives from an empty interaction dataset."
        raise ValueError(msg)

    if item_count < 1:
        msg = "item_count must be at least 1."
        raise ValueError(msg)

    if not positive_interactions["target"].eq(POSITIVE_TARGET).all():
        msg = "Negative sampling input must contain only positive targets."
        raise ValueError(msg)

    if positive_interactions["item_idx"].lt(0).any():
        msg = "item_idx cannot contain negative values."
        raise ValueError(msg)

    if positive_interactions["item_idx"].ge(item_count).any():
        msg = "item_idx contains values outside the known item range."
        raise ValueError(msg)


def _validate_required_columns(interactions: pd.DataFrame) -> None:
    """Validate required encoded-interaction columns."""
    missing_columns = REQUIRED_COLUMNS - set(interactions.columns)

    if missing_columns:
        missing = ", ".join(sorted(missing_columns))
        msg = f"Missing required negative sampling columns: {missing}."
        raise ValueError(msg)


def _validate_no_positive_overlap(
    negatives: pd.DataFrame,
    positive_pairs: set[tuple[int, int]],
) -> None:
    """Ensure that generated negatives are not known positives."""
    negative_pairs = set(
        zip(
            negatives["user_idx"].astype(int),
            negatives["item_idx"].astype(int),
            strict=True,
        )
    )

    overlap = positive_pairs & negative_pairs

    if overlap:
        msg = "Generated negative samples overlap positive interactions."
        raise RuntimeError(msg)
