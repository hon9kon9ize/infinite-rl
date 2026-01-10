"""Infinite RL - Generate synthetic RL datasets for LLM preference optimization."""

from src.executor import RewardExecutor
from src.reward_functions import get_reward_functions

__version__ = "0.1"
__all__ = ["RewardExecutor", "get_reward_functions"]
