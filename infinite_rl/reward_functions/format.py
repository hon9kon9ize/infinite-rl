import re
from typing import Union, TYPE_CHECKING
from .reward_function import RewardFunction, RewardFunctionScore

if TYPE_CHECKING:
    from ..task import Task


class FormatRewardFunction(RewardFunction):
    """Reward function that evaluates only formatting of the model response.

    Behavior:
    - All tasks must have content wrapped in <answer> tags.
    - Returns 1.0 if <answer> tag is present and has non-empty content (code blocks
      are automatically extracted), 0.0 otherwise.
    - This validator ensures consistent formatting across all task types.
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
        task: "Task",
        **kwargs,
    ) -> RewardFunctionScore:
        if not self.initialized:
            self.initialize()

        # First extract tag content - don't strip code blocks yet
        tag_start = f"<{self.target_tag}>"
        tag_end = f"</{self.target_tag}>"

        import re

        pattern = f"{re.escape(tag_start)}(.*?){re.escape(tag_end)}"
        matches = re.findall(pattern, task.model_output or "", re.DOTALL)

        if not matches:
            return RewardFunctionScore(
                score=0.0,
                info=f"No content found in the <{self.target_tag}> tag.",
            )

        raw_content = "\n".join(matches)

        # For math tasks, check if content has code blocks (should not)
        if self.task_name == "math":
            if "```" in raw_content:
                return RewardFunctionScore(
                    score=0.0,
                    info=f"Math task should not contain code blocks.",
                )
            # Math should have non-empty content without code blocks
            if raw_content.strip():
                return RewardFunctionScore(score=1.0, info="Valid math answer.")
            else:
                return RewardFunctionScore(score=0.0, info="Empty math answer.")

        # For code/puzzle tasks, check proper code block formatting
        if "```" in raw_content:
            # Count opening and closing backticks
            opening_count = raw_content.count("```")
            # Check if we have balanced code blocks (opening + closing pairs)
            if opening_count % 2 != 0:
                # Odd number of triple-backtick sequences = malformed
                return RewardFunctionScore(
                    score=0.0,
                    info=f"Code block not properly closed (missing closing ```).",
                )
            # Check if code blocks have language specifier
            code_pattern = r"```(\w+)"
            has_language = bool(re.search(code_pattern, raw_content))
            if has_language:
                return RewardFunctionScore(
                    score=1.0, info="Valid code block with language specifier."
                )
            else:
                # Code blocks present but no language specifier - still valid
                return RewardFunctionScore(
                    score=1.0, info="Valid code block (no language specifier)."
                )

        # No code blocks - check if content is non-empty
        if raw_content.strip():
            return RewardFunctionScore(score=1.0, info="Valid answer tag with content.")
        else:
            return RewardFunctionScore(score=0.0, info="Empty answer tag.")
