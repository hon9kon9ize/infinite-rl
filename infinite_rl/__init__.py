"""Infinite RL - Generate synthetic RL datasets for LLM preference optimization."""

from .executor import Executor
from .reward_functions import get_reward_functions
from .run_examples import run_examples

__version__ = "0.1.13"
__all__ = ["Executor", "get_reward_functions", "run_examples"]
