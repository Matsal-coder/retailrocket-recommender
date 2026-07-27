"""Tests for implicit-feedback negative sampling."""

import pandas as pd
import pytest

from retail_recommender.features.negative_sampling import (
    NEGATIVE_TARGET,
    NegativeSamplingConfig,
    combine_positive_and_negative_interactions,
    generate_negative_samples,
)

USER_A = 0
USER_B = 1

ITEM_A = 0
ITEM_B = 1
ITEM_C = 2
ITEM_D = 3
ITEM_E = 4

ITEM_COUNT = 5
RANDOM_SEED = 1729
NEGATIVES_PER_POSITIVE = 2

EXPECTED_POSITIVE_COUNT = 3
EXPECTED_NEGATIVE_COUNT = EXPECTED_POSITIVE_COUNT * NEGATIVES_PER_POSITIVE
EXPECTED_COMBINED_COUNT = EXPECTED_POSITIVE_COUNT + EXPECTED_NEGATIVE_COUNT


def make_positive_interactions() -> pd.DataFrame:
    """Create encoded positive interactions for sampling tests."""
    return pd.DataFrame(
        {
            "user_idx": [USER_A, USER_A, USER_B],
            "item_idx": [ITEM_A, ITEM_B, ITEM_C],
            "target": [1, 1, 1],
        }
    )


def make_config(
    negative_samples_per_positive: int = NEGATIVES_PER_POSITIVE,
    random_seed: int = RANDOM_SEED,
) -> NegativeSamplingConfig:
    """Create default negative sampling settings."""
    return NegativeSamplingConfig(
        negative_samples_per_positive=negative_samples_per_positive,
        random_seed=random_seed,
    )


def test_generate_negative_samples_creates_expected_count() -> None:
    """It should generate the configured negatives per positive."""
    positives = make_positive_interactions()

    result = generate_negative_samples(
        positive_interactions=positives,
        item_count=ITEM_COUNT,
        config=make_config(),
    )

    assert len(result) == EXPECTED_NEGATIVE_COUNT
    assert result["target"].eq(NEGATIVE_TARGET).all()


def test_generate_negative_samples_does_not_overlap_positives() -> None:
    """It should never sample a known positive pair."""
    positives = make_positive_interactions()

    negatives = generate_negative_samples(
        positive_interactions=positives,
        item_count=ITEM_COUNT,
        config=make_config(),
    )

    positive_pairs = set(
        zip(
            positives["user_idx"],
            positives["item_idx"],
            strict=True,
        )
    )
    negative_pairs = set(
        zip(
            negatives["user_idx"],
            negatives["item_idx"],
            strict=True,
        )
    )

    assert positive_pairs.isdisjoint(negative_pairs)


def test_generate_negative_samples_is_reproducible() -> None:
    """It should produce the same output with the same seed."""
    positives = make_positive_interactions()
    config = make_config()

    first = generate_negative_samples(
        positive_interactions=positives,
        item_count=ITEM_COUNT,
        config=config,
    )
    second = generate_negative_samples(
        positive_interactions=positives,
        item_count=ITEM_COUNT,
        config=config,
    )

    pd.testing.assert_frame_equal(first, second)


def test_generate_negative_samples_preserves_user_counts() -> None:
    """Each user should receive negatives proportional to positives."""
    positives = make_positive_interactions()

    negatives = generate_negative_samples(
        positive_interactions=positives,
        item_count=ITEM_COUNT,
        config=make_config(),
    )

    positive_counts = positives.groupby("user_idx").size()
    negative_counts = negatives.groupby("user_idx").size()

    expected_counts = positive_counts * NEGATIVES_PER_POSITIVE

    pd.testing.assert_series_equal(
        negative_counts,
        expected_counts,
        check_names=False,
    )


def test_generate_negative_samples_allows_replacement_when_needed() -> None:
    """It should sample with replacement when unseen items are scarce."""
    positives = pd.DataFrame(
        {
            "user_idx": [USER_A, USER_A, USER_A, USER_A],
            "item_idx": [ITEM_A, ITEM_B, ITEM_C, ITEM_D],
            "target": [1, 1, 1, 1],
        }
    )

    negatives = generate_negative_samples(
        positive_interactions=positives,
        item_count=ITEM_COUNT,
        config=make_config(),
    )

    assert len(negatives) == len(positives) * NEGATIVES_PER_POSITIVE
    assert negatives["item_idx"].eq(ITEM_E).all()


def test_generate_negative_samples_rejects_user_with_all_items() -> None:
    """It should reject users without available negative items."""
    positives = pd.DataFrame(
        {
            "user_idx": [USER_A] * ITEM_COUNT,
            "item_idx": list(range(ITEM_COUNT)),
            "target": [1] * ITEM_COUNT,
        }
    )

    with pytest.raises(ValueError, match="every known item"):
        generate_negative_samples(
            positive_interactions=positives,
            item_count=ITEM_COUNT,
            config=make_config(),
        )


def test_generate_negative_samples_rejects_non_positive_input() -> None:
    """It should require a positive-only input dataset."""
    invalid = make_positive_interactions()
    invalid.loc[0, "target"] = NEGATIVE_TARGET

    with pytest.raises(ValueError, match="only positive targets"):
        generate_negative_samples(
            positive_interactions=invalid,
            item_count=ITEM_COUNT,
            config=make_config(),
        )


def test_generate_negative_samples_rejects_missing_columns() -> None:
    """It should require encoded user, item and target columns."""
    invalid = pd.DataFrame(
        {
            "user_idx": [USER_A],
            "target": [1],
        }
    )

    with pytest.raises(ValueError, match="item_idx"):
        generate_negative_samples(
            positive_interactions=invalid,
            item_count=ITEM_COUNT,
            config=make_config(),
        )


def test_generate_negative_samples_rejects_out_of_range_items() -> None:
    """It should reject item indices outside the known range."""
    invalid = make_positive_interactions()
    invalid.loc[0, "item_idx"] = ITEM_COUNT

    with pytest.raises(ValueError, match="outside the known item range"):
        generate_negative_samples(
            positive_interactions=invalid,
            item_count=ITEM_COUNT,
            config=make_config(),
        )


def test_combine_positive_and_negative_interactions() -> None:
    """It should combine both target classes into one training dataset."""
    positives = make_positive_interactions()
    negatives = generate_negative_samples(
        positive_interactions=positives,
        item_count=ITEM_COUNT,
        config=make_config(),
    )

    combined = combine_positive_and_negative_interactions(
        positive_interactions=positives,
        negative_interactions=negatives,
        random_seed=RANDOM_SEED,
    )

    assert len(combined) == EXPECTED_COMBINED_COUNT
    assert set(combined["target"].unique()) == {
        0,
        1,
    }


def test_negative_sampling_config_rejects_invalid_ratio() -> None:
    """It should reject ratios below one."""
    with pytest.raises(ValueError, match="at least 1"):
        NegativeSamplingConfig(
            negative_samples_per_positive=0,
            random_seed=RANDOM_SEED,
        )
