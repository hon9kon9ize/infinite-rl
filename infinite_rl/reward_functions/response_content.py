"""Response content length reward.

Rewards having a brief explanation/summary between the reasoning section
(`</think>`) and the answer section (`<answer>`). Encourages a "sweet spot"
of substance without verbosity.
"""
import math
from typing import Optional, Any, TYPE_CHECKING
from .reward_function import RewardFunction, RewardFunctionScore

if TYPE_CHECKING:
    from ..task import Task


def response_content_length_reward(
    length: int,
    min_len: int = 10,
    sweet_start: int = 30,
    sweet_end: int = 500,
    max_len: int = 1000,
) -> float:
    """Score response content length with a sweet-spot curve.

    Regions:
    - 0: No content -> 0.0
    - [0, min_len): Too short -> linear ramp 0.0 -> 0.3
    - [min_len, sweet_start): Acceptable -> linear ramp 0.3 -> 1.0
    - [sweet_start, sweet_end): Sweet spot -> 1.0
    - (sweet_end, max_len]: Too verbose -> cosine decay 1.0 -> 0.4
    - > max_len: Very verbose -> 0.4
    """
    if length == 0:
        return 0.0

    if length < min_len:
        # Too short — linear ramp from 0.0 to 0.3
        return max(0.0, 0.3 * length / min_len)

    if length < sweet_start:
        # Approaching sweet spot — linear ramp from 0.3 to 1.0
        frac = (length - min_len) / (sweet_start - min_len)
        return 0.3 + 0.7 * frac

    if length <= sweet_end:
        # Sweet spot
        return 1.0

    if length <= max_len:
        # Gentle cosine decay from 1.0 to 0.4
        x = (length - sweet_end) / (max_len - sweet_end)
        x = max(0.0, min(1.0, x))
        decay = (math.cos(math.pi * x) + 1.0) / 2.0  # 1.0 -> 0.0
        return 0.4 + 0.6 * decay

    # Very verbose
    return 0.4


class ResponseContentRewardFunction(RewardFunction):
    """Reward function that scores the length of content between </think> and <answer>.

    Encourages a brief explanation/summary after reasoning and before the answer tag.
    Uses a sweet-spot curve: full reward for 30-500 chars, penalties for empty or verbose.

    Returns an auxiliary signal in `score` (0..1).
    """

    def __init__(
        self,
        task_name: str = "response_content",
        timeout: int = 5,
        answer_tag: str = "answer",
        think_tag: str = "think",
        reasoning_template: bool = False,
        min_len: int = 10,
        sweet_start: int = 30,
        sweet_end: int = 500,
        max_len: int = 1000,
        require_non_empty_think: bool = True,
        **kwargs,
    ):
        super().__init__(
            task_name,
            timeout=timeout,
            answer_tag=answer_tag,
            think_tag=think_tag,
            reasoning_template=reasoning_template,
        )
        self.min_len = min_len
        self.sweet_start = sweet_start
        self.sweet_end = sweet_end
        self.max_len = max_len
        self.require_non_empty_think = require_non_empty_think

    def initialize(self):
        self.initialized = True

    def _extract_response_content(self, model_output: str) -> str:
        """Extract content between </think> and <answer>.

        For reasoning_template mode, the opening <think> is omitted by chat template,
        so we look for </think> -> <answer>.
        For standard mode, we look for </think> -> <answer>.
        """
        output = model_output or ""
        think_close = f"</{self.think_tag}>"
        answer_open = f"<{self.answer_tag}>"

        think_end = output.find(think_close)
        if think_end < 0:
            return ""

        # Start after </think>
        start = think_end + len(think_close)

        answer_start = output.find(answer_open, start)
        if answer_start < 0:
            # No <answer> tag — return everything after </think>
            return output[start:].strip()

        return output[start:answer_start].strip()

    def compute_reward(
        self,
        task: "Task",
        **kwargs,
    ) -> RewardFunctionScore:
        """Compute response content length reward."""
        if not self.initialized:
            self.initialize()

        if self.require_non_empty_think:
            think_content = self.extract_think_content(
                task.model_output or "",
                tag=self.think_tag,
            )
            if not think_content.strip():
                return RewardFunctionScore(
                    score=0.0,
                    info="No response_content reward because reasoning content is empty.",
                )

        content = self._extract_response_content(task.model_output or "")

        length = len(content)

        reward = response_content_length_reward(
            length,
            min_len=self.min_len,
            sweet_start=self.sweet_start,
            sweet_end=self.sweet_end,
            max_len=self.max_len,
        )

        return RewardFunctionScore(
            score=float(reward),
            info=f"response_content: {length} chars" if length > 0 else "No content between </think> and <answer>.",
        )
