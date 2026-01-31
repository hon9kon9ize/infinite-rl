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
        target_tag: str = None,
    ):
        super().__init__(
            task_name,
            timeout=timeout,
            answer_tag=answer_tag,
            think_tag=think_tag,
            target_tag=target_tag,
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

        # count how many tag_start and tag_end are present
        start_count = task.model_output.count(tag_start)
        end_count = task.model_output.count(tag_end)

        if start_count > 1 or end_count > 1:
            return RewardFunctionScore(
                score=-1.0,
                info=f"Multiple <{self.target_tag}> tags found.",
            )

        if not matches:
            return RewardFunctionScore(
                score=-1.0,
                info=f"No content found in the <{self.target_tag}> tag.",
            )

        # Check if there's content before the opening tag (format violation)
        tag_start_index = task.model_output.find(tag_start)
        content_before_tag = task.model_output[:tag_start_index].strip()

        # For think tag: must be at the very start (no content before)
        # For answer tag: allow think tag before it, but no other content
        if self.target_tag == self.think_tag:
            # Think tag must be first - no content allowed before it
            if content_before_tag:
                return RewardFunctionScore(
                    score=-1.0,
                    info=f"Content found before <{self.target_tag}> opening tag. Tags must appear at the start.",
                )
        elif self.target_tag == self.answer_tag:
            # Answer tag can have think tag before it, but check for other content
            # Remove the think tag section if present
            if content_before_tag:
                think_tag_pattern = f"<{self.think_tag}>.*?</{self.think_tag}>"
                content_without_think = re.sub(
                    think_tag_pattern, "", content_before_tag, flags=re.DOTALL
                ).strip()
                if content_without_think:
                    return RewardFunctionScore(
                        score=-1.0,
                        info=f"Content found before <{self.target_tag}> opening tag (excluding valid <{self.think_tag}> section).",
                    )

        raw_content = "\n".join(matches)

        # Check for nested/misplaced tags: answer tag should not appear inside think tag and vice versa
        # Determine the other tag based on current target
        if self.target_tag == self.think_tag:
            forbidden_tag = self.answer_tag
        elif self.target_tag == self.answer_tag:
            forbidden_tag = self.think_tag
        else:
            forbidden_tag = None

        # Check if the forbidden tag appears inside the current tag's content
        if forbidden_tag:
            forbidden_start = f"<{forbidden_tag}>"
            forbidden_end = f"</{forbidden_tag}>"
            if forbidden_start in raw_content or forbidden_end in raw_content:
                return RewardFunctionScore(
                    score=-1.0,
                    info=f"<{forbidden_tag}> tag found inside <{self.target_tag}> tag. Tags cannot be nested.",
                )

        # For math tasks, check if content has code blocks (should not)
        if self.task_name == "math":
            if "```" in raw_content:
                return RewardFunctionScore(
                    score=-1.0,
                    info=f"Math task should not contain code blocks.",
                )
            # Math should have non-empty content without code blocks
            if raw_content.strip():
                return RewardFunctionScore(score=1.0, info="Valid math answer.")
            else:
                return RewardFunctionScore(score=-1.0, info="Empty math answer.")

        # For code/puzzle tasks, check proper code block formatting
        if "```" in raw_content:
            # Count opening and closing backticks
            opening_count = raw_content.count("```")
            # Check if we have balanced code blocks (opening + closing pairs)
            if opening_count % 2 != 0:
                # Odd number of triple-backtick sequences = malformed
                return RewardFunctionScore(
                    score=-1.0,
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
            return RewardFunctionScore(score=-1.0, info="Empty answer tag.")
