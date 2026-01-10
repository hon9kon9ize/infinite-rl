"""Infinite RL - Generate synthetic RL datasets for LLM preference optimization."""

from .executor import RewardExecutor
from .reward_functions import get_reward_functions

__version__ = "0.1"
__all__ = ["RewardExecutor", "get_reward_functions"]
