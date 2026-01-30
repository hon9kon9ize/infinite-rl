import re
from typing import Union, TYPE_CHECKING
from .reward_function import RewardFunction, RewardFunctionScore

if TYPE_CHECKING:
    from ..task import Task


class ReasoningStepsRewardFunction(RewardFunction):
    """Reward function that gives bonuses/penalties based on reasoning step indicators.

    The function looks for a <think> tag and checks for presence of common
    step/analysis indicators ("first", "second", "finally", "therefore", etc.).

    Returns:
    - 0.7: Two or more indicators found (strong reasoning)
    - 0.5: One indicator found (basic reasoning)
    - -1.0: No indicators found (penalty for lack of structured reasoning)
    """

    def __init__(
        self,
        task_name: str = "reasoning_steps",
        timeout: int = 5,
        target_tag: str = "think",
        **kwargs,
    ):
        super().__init__(task_name, timeout=timeout, target_tag=target_tag, **kwargs)

    def initialize(self):
        self.initialized = True

    def compute_reward(
        self,
        task: "Task",
        **kwargs,
    ) -> RewardFunctionScore:
        # Ensure initialized
        if not self.initialized:
            self.initialize()

        if not task.model_output:
            return RewardFunctionScore(score=0.0)

        # Extract think content using the think_tag
        think_content = self.extract_tag(task.model_output)
        if think_content:
            thinking_content = think_content.lower()
        else:
            return RewardFunctionScore(
                score=0.0,
                info=f"No <{self.think_tag}> tag content found.",
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

        if found_count >= 2:
            bonus = 0.7
        elif found_count > 0:
            bonus = 0.5
        else:
            bonus = -1.0

        # Signal the bonus in the single score field
        return RewardFunctionScore(score=float(bonus))
