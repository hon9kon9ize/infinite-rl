import math
import pytest
from infinite_rl.reward_functions.length import reasoning_friendly_length_reward


def test_short_penalty():
    # Less than 100 chars -> 0.1
    assert reasoning_friendly_length_reward(10, target_len=800, max_len=1000) == 0.1
    assert reasoning_friendly_length_reward(99, target_len=800, max_len=1000) == 0.1


def test_sweet_spot():
    # 100 to target_len (800) -> 1.0
    assert reasoning_friendly_length_reward(100, target_len=800, max_len=1000) == 1.0
    assert reasoning_friendly_length_reward(400, target_len=800, max_len=1000) == 1.0
    assert reasoning_friendly_length_reward(799, target_len=800, max_len=1000) == 1.0


def test_rambling_decay():
    # At max_len (1000) -> 0.5
    # x = (1000-800)/200 = 1.0
    # cos(pi) = -1
    # (-1+1)/4 + 0.5 = 0.5
    r = reasoning_friendly_length_reward(1000, target_len=800, max_len=1000)
    assert pytest.approx(0.5, abs=1e-6) == r

    # Halfway through decay zone (900)
    # x = (900-800)/200 = 0.5
    # cos(pi/2) = 0
    # (0+1)/4 + 0.5 = 0.25 + 0.5 = 0.75
    r_mid = reasoning_friendly_length_reward(900, target_len=800, max_len=1000)
    assert pytest.approx(0.75, abs=1e-6) == r_mid


def test_default_max_len():
    # Default max_len is 3584, target_len approx 0.8*3584 = 2867
    target_len = 2867
    assert reasoning_friendly_length_reward(2000, target_len=target_len) == 1.0
    assert reasoning_friendly_length_reward(3584, target_len=target_len) == 0.5
