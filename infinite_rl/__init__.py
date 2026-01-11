"""Infinite RL - Generate synthetic RL datasets for LLM preference optimization."""

from .executor import RewardExecutor
from .reward_functions import get_reward_functions
from .run_examples import run_examples

__version__ = "0.1.7"
__all__ = ["RewardExecutor", "get_reward_functions", "run_examples"]
