"""Length-based reward utilities.

Provides Cosine Length Reward to penalize verbosity or laziness depending on
whether the model was correct.
"""

import math
from typing import Optional


def cosine_length_reward(
    length: int,
    min_len: int = 1,
    max_len: int = 1000,
    target_len: Optional[int] = None,
    correct: bool = True,
) -> float:
    """Compute a length-based reward in [0, 1].

    Behavior:
      - When `correct` is True: short/economical answers are preferred up to
        `target_len`. For lengths <= target_len the reward is 1.0. Beyond
        `target_len` it smoothly decays to 0 at `max_len` using a cosine curve.
      - When `correct` is False: short answers are penalized (lazy); reward
        increases with length (encourage effort) using a cosine curve that maps
        short -> 0 and long -> 1.

    Args:
        length: response length (tokens or words) to score.
        min_len: minimum length considered (defaults to 1).
        max_len: maximum length (cap) (defaults to 1000).
        target_len: for correct=True, the target/"sweet-spot" upper bound.
                    If None, defaults to min_len (i.e., shorter is always better).
        correct: whether the answer is correct (affects curve direction).

    Returns:
        A float in [0.0, 1.0].
    """
    # Basic sanitization
    if max_len <= min_len:
        raise ValueError("max_len must be greater than min_len")

    L = max(min_len, min(length, max_len))

    if correct:
        # If no target provided use the most economical policy (shorter is better)
        target = (
            min_len if target_len is None else max(min_len, min(target_len, max_len))
        )

        if L <= target:
            return 1.0

        denom = max_len - target
        if denom <= 0:
            # Edge: target == max_len
            return 0.0

        x = (L - target) / denom  # 0..1
        # Map cos(pi * x) from [-1,1] to [0,1] with peak at x=0
        return float((math.cos(math.pi * x) + 1.0) / 2.0)
    else:
        # Incorrect: encourage longer attempts
        denom = max_len - min_len
        if denom <= 0:
            return 0.0
        x = (L - min_len) / denom  # 0..1
        # Use (1 - cos(pi * x)) / 2 which maps x=0 -> 0, x=1 -> 1, smooth curve
        return float((1.0 - math.cos(math.pi * x)) / 2.0)


# New: LengthRewardFunction – wraps cosine_length_reward into a RewardFunction
from typing import Union
from .reward_function import RewardFunction, RewardFunctionScore
from ..utils.parser_utils import extract_answer_tags


class LengthRewardFunction(RewardFunction):
    """Reward function that scores response length using `cosine_length_reward`.

    expected_output may be an integer indicating the target length (in tokens/words).

    Returns an auxiliary signal in `score` (0..1). This reward is aux-only (it does not check correctness).
    """

    def __init__(
        self,
        task_name: str = "length",
        timeout: int = 5,
        min_len: int = 1,
        max_len: int = 1000,
        target_len: Optional[int] = None,
        answer_tag: str = "answer",
        think_tag: str = "think",
    ):
        super().__init__(
            task_name, timeout=timeout, answer_tag=answer_tag, think_tag=think_tag
        )
        self.min_len = min_len
        self.max_len = max_len
        self.target_len = target_len

    def initialize(self):
        self.initialized = True

    def _extract_length(self, model_output: str, answer_tag: str = None) -> int:
        answer_tag = answer_tag or self.answer_tag
        matches = extract_answer_tags(model_output, tag=answer_tag)
        text = matches if matches else model_output
        tokens = text.strip().split()
        return max(0, len(tokens))

    def compute_reward(
        self,
        model_output: str,
        expected_output: Union[str, int, None],
        is_correct: Optional[bool] = None,
    ) -> RewardFunctionScore:
        """Compute length reward.

        Parameters:
          - expected_output: optional numeric target length (int), else uses configured target.
          - is_correct: optional boolean indicating whether the main task was correct. If None,
            defaults to True. When True, shorter answers are preferred; when False,
            longer answers are preferred.
        """
        if not self.initialized:
            self.initialize()

        length = self._extract_length(model_output)
        # Determine target length: prefer explicit expected_output if numeric, else fallback to configured target
        target = None
        if expected_output is not None:
            try:
                target = int(expected_output)
            except Exception:
                target = None
        if target is None:
            target = self.target_len

        correct_flag = True if is_correct is None else bool(is_correct)

        len_reward = cosine_length_reward(
            length,
            min_len=self.min_len,
            max_len=self.max_len,
            target_len=target,
            correct=correct_flag,
        )

        # Aux-only behavior: return the length reward in the single score field
        return RewardFunctionScore(score=float(len_reward))
