"""Infinite RL - Generate synthetic RL datasets for LLM preference optimization."""

from .executor import Executor
from .reward_functions import get_reward_functions
from .run_examples import run_examples

# Expose package version from VERSION.txt included in the package
try:
    # Preferred: use importlib.resources (works on installed packages)
    try:
        from importlib import resources

        __version__ = resources.read_text(__package__, "VERSION.txt").strip()
    except Exception:
        # Fallback to reading from repo root (useful in dev checkouts)
        import os

        root_version = os.path.abspath(
            os.path.join(os.path.dirname(__file__), "..", "VERSION.txt")
        )
        try:
            with open(root_version, "r") as f:
                __version__ = f.read().strip()
        except Exception:
            __version__ = "0.0.0"
except Exception:
    __version__ = "0.0.0"

__all__ = ["Executor", "get_reward_functions", "run_examples", "RewardOrchestrator"]

# Import orchestration helper
from .reward_orchestrator import RewardOrchestrator
