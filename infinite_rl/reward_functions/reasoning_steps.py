import re
from typing import Union
from .reward_function import RewardFunction, RewardFunctionScore
from ..utils.parser_utils import extract_tag


class ReasoningStepsRewardFunction(RewardFunction):
    """Reward function that gives a small bonus when the model provides explicit
    reasoning steps inside a <think>...</think> block.

    The function looks for a <think> tag and checks for presence of common
    step/analysis indicators ("first", "second", "finally", "therefore", etc.).
    It returns a small encouragement bonus (0.1 or 0.2) as score.
    """

    def __init__(
        self,
        task_name: str = "reasoning_steps",
        timeout: int = 5,
        answer_tag: str = "answer",
        think_tag: str = "think",
    ):
        super().__init__(
            task_name, timeout=timeout, answer_tag=answer_tag, think_tag=think_tag
        )

    def initialize(self):
        self.initialized = True

    def compute_reward(
        self,
        model_output: str,
        **kwargs,
    ) -> RewardFunctionScore:
        # Ensure initialized
        if not self.initialized:
            self.initialize()

        if not model_output:
            return RewardFunctionScore(score=0.0)

        # Extract think content using the think_tag
        think_content = self.extract_tag(model_output, target_tag=self.think_tag)
        if think_content:
            thinking_content = think_content.lower()
        else:
            return RewardFunctionScore(
                score=0.0,
                error_msg={
                    "reasoning_steps": f"No <{self.think_tag}> tag content found."
                },
            )

        indicators = [
            "step",
            "first",
            "second",
            "third",
            "finally",
            "therefore",
            "thus",
            "consequently",
            "wait",
            "let me",
            "re-check",
            "however",
        ]

        found_count = sum(1 for word in indicators if word in thinking_content)

        if found_count >= 3:
            bonus = 0.2
        elif found_count > 0:
            bonus = 0.1
        else:
            bonus = 0.0

        # Signal the bonus in the single score field
        return RewardFunctionScore(score=float(bonus))
