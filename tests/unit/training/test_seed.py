"""Unit tests for reproducibility utilities."""

import random

import numpy as np
import pytest
import torch

from retail_recommender.training.seed import set_global_seed

TEST_SEED = 731


def test_global_seed_reproduces_random_values() -> None:
    set_global_seed(TEST_SEED)

    first_python_value = random.random()
    first_numpy_value = np.random.random()
    first_torch_value = torch.rand(1)

    set_global_seed(TEST_SEED)

    assert random.random() == first_python_value
    assert np.random.random() == first_numpy_value
    assert torch.equal(torch.rand(1), first_torch_value)


def test_global_seed_rejects_negative_value() -> None:
    with pytest.raises(
        ValueError,
        match="seed must be non-negative",
    ):
        set_global_seed(-1)
