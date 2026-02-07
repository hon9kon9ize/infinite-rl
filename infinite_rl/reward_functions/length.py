"""Length-based reward utilities.

Provides Reasoning Friendly Length Reward to encourage appropriate length
without penalizing correctness or rewarding rambling (arXiv:2503.16219v2).
"""

import math
from typing import Optional, Any, TYPE_CHECKING
from .reward_function import RewardFunction, RewardFunctionScore

if TYPE_CHECKING:
    from ..task import Task


def reasoning_friendly_length_reward(
    length: int, target_len: int, max_len: int = 3584
) -> float:
    """
    Following arXiv:2503.16219v2:
    Don't punish length for correct answers.
    Don't reward rambling for wrong answers.

    The "sweet spot" plateau is from 100 to target_len characters.
    Beyond target_len, a gentle cosine decay to 0.5 at max_len.
    """

    # 1. Minimum Effort Floor (Characters)
    if length < 100:
        return 0.1  # Very low reward for 'lazy' thinking

    # 2. The "Sweet Spot" (The Plateau)
    # Between 100 and target_len, the reward is 1.0 (Neutral)
    if length < target_len:
        return 1.0

    # 3. The "Rambling" Penalty (Beyond target_len)
    # Gentle decay from target_len to max_len
    denom = max_len - target_len
    if denom <= 0:  # Safety
        return 0.5

    x = (length - target_len) / denom
    x = max(0.0, min(float(x), 1.0))

    # Cosine decay from 1.0 down to 0.5 (never 0.0)
    decay = (math.cos(math.pi * x) + 1.0) / 4.0 + 0.5
    return float(decay)


class LengthRewardFunction(RewardFunction):
    """Reward function that scores response length using `reasoning_friendly_length_reward`.

    Uses level-specific target lengths for the plateau:
    - Level 0: 1500
    - Level 1: 1500
    - Level 2: 2000
    - Levels 3-6: 3000

    Returns an auxiliary signal in `score` (0..1).
    """

    def __init__(
        self,
        task_name: str = "length",
        timeout: int = 5,
        max_len: int = 3584,
        **kwargs,
    ):
        # Remove legacy arguments if present in kwargs to avoid issues
        kwargs.pop("min_len", None)
        kwargs.pop("target_len", None)

        super().__init__(task_name, timeout=timeout, **kwargs)
        self.max_len = max_len
        # Ensure target_tag is set if not provided but think_tag is
        if not hasattr(self, "target_tag") or not self.target_tag:
            self.target_tag = kwargs.get("think_tag", "think")

    def initialize(self):
        self.initialized = True

    def compute_reward(
        self,
        task: "Task",
        **kwargs,
    ) -> RewardFunctionScore:
        """Compute length reward."""
        if not self.initialized:
            self.initialize()

        thought_content = self.extract_tag(task.model_output or "")

        if not thought_content:
            return RewardFunctionScore(
                score=0.0,
                info=f"No content found in the <{self.target_tag}> tag.",
            )

        length = len(thought_content.strip())

        # Get target length based on task level
        target_lengths = {0: 1500, 1: 1500, 2: 2000, 3: 2000, 4: 3000, 5: 3000, 6: 3000}
        target_len = target_lengths.get(task.level, 2000)

        len_reward = reasoning_friendly_length_reward(
            length,
            target_len=target_len,
            max_len=self.max_len,
        )

        return RewardFunctionScore(score=float(len_reward))
