import re
from typing import Union
from .reward_function import RewardFunction, RewardFunctionScore
from ..utils.parser_utils import extract_tag


class FormatRewardFunction(RewardFunction):
    """Reward function that evaluates only formatting of the model response.

    Behavior:
    - For code tasks (python/javascript): expects a single <answer> tag containing a fenced code block.
      Returns 1.0 for properly fenced block with language tag, 0.5 for fenced block without language tag,
      0.2 for malformed fenced block, and 0.0 for missing <answer> tag.
    - For math tasks: expects a single <answer> tag containing a simple value (no backticks).
      Returns 1.0 for a single tag with a non-empty value, 0.0 otherwise.
    - For other tasks, as a generic fallback: 1.0 if <answer> tag present and non-empty, 0.0 otherwise.
    """

    def __init__(
        self,
        task_name: str = "format",
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
        if not self.initialized:
            self.initialize()

        # Use the answer_tag by default
        target_tag = kwargs.get("target_tag", self.answer_tag)
        content = self.extract_tag(model_output, target_tag=target_tag)

        if not content:
            return RewardFunctionScore(
                score=0.0,
                error_msg={"format": f"No content found in the <{target_tag}> tag."},
            )

        # For code tasks (python/javascript), check for code blocks
        if self.task_name in ["python", "javascript"]:
            # Look for properly closed code blocks with language specification
            code_block_pattern = r"```(\w+)\n(.*?)```"
            matches = re.findall(code_block_pattern, content, re.DOTALL)

            if matches:
                # Has properly closed code blocks with language tags
                return RewardFunctionScore(score=1.0)

            # Check for improperly formatted code blocks
            if "```" in content:
                # Has backticks but not properly closed
                return RewardFunctionScore(score=0.0)

            # No code block found
            return RewardFunctionScore(score=0.0)
        elif self.task_name == "math":
            if re.search(r"```", content):
                # Math shouldn't have code blocks
                return RewardFunctionScore(score=0.0)
            else:
                # Simple value, no code blocks
                return RewardFunctionScore(score=1.0)

        # Generic fallback for other tasks
        else:
            return RewardFunctionScore(score=1.0 if content else 0.0)
