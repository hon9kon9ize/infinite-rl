import re
from typing import Union
from .reward_function import RewardFunction, RewardFunctionScore


class ReasoningStepsRewardFunction(RewardFunction):
    """Reward function that gives a small bonus when the model provides explicit
    reasoning steps inside a <think>...</think> block.

    The function looks for a <think> tag and checks for presence of common
    step/analysis indicators ("first", "second", "finally", "therefore", etc.).
    It returns a small encouragement bonus (0.1 or 0.2) as correctness_score while
    using format_score to indicate whether a <think> block was present.
    """

    def __init__(self, task_name: str = "reasoning_steps", timeout: int = 5):
        super().__init__(task_name, timeout=timeout)

    def initialize(self):
        self.initialized = True

    def compute_reward(
        self,
        model_output: str,
        expected_output: Union[str, int, None],
        answer_tag: str = "answer",
    ) -> RewardFunctionScore:
        # Ensure initialized
        if not self.initialized:
            self.initialize()

        if not model_output:
            return RewardFunctionScore(format_score=0.0, correctness_score=0.0)

        # Extract <think> block
        m = re.search(r"<think>(.*?)</think>", model_output, re.DOTALL | re.IGNORECASE)
        if not m:
            return RewardFunctionScore(
                format_score=0.0,
                correctness_score=0.0,
                error_msg="Missing <think> tags in response.",
            )

        thinking_content = m.group(1).lower()

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

        return RewardFunctionScore(format_score=1.0, correctness_score=bonus)
