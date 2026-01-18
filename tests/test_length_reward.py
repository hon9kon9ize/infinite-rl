import math
import pytest
from infinite_rl.reward_functions.length import cosine_length_reward


def test_correct_short_preferred():
    # Below target -> full reward
    assert (
        cosine_length_reward(10, min_len=1, max_len=1000, target_len=200, correct=True)
        == 1.0
    )


def test_correct_long_penalty():
    # At max -> reward should be zero
    r = cosine_length_reward(
        1000, min_len=1, max_len=1000, target_len=200, correct=True
    )
    assert pytest.approx(0.0, abs=1e-6) == r


def test_correct_mid_decay():
    # length halfway between target and max -> expect about cos(pi*0.5)->0 mapped to 0.5
    mid = (200 + 1000) // 2
    r = cosine_length_reward(mid, min_len=1, max_len=1000, target_len=200, correct=True)
    assert pytest.approx(0.5, rel=1e-2) == r


def test_incorrect_short_penalized():
    r = cosine_length_reward(1, min_len=1, max_len=1000, target_len=200, correct=False)
    assert pytest.approx(0.0, abs=1e-6) == r


def test_incorrect_long_encouraged():
    r = cosine_length_reward(1000, min_len=1, max_len=1000, correct=False)
    assert pytest.approx(1.0, abs=1e-6) == r


def test_invalid_bounds():
    with pytest.raises(ValueError):
        cosine_length_reward(10, min_len=100, max_len=50, correct=True)
