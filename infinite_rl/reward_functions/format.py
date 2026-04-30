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
        reasoning_template: bool = False,
        allow_explanation_between_tags: bool = True,
    ):
        super().__init__(
            task_name,
            timeout=timeout,
            answer_tag=answer_tag,
            think_tag=think_tag,
            target_tag=target_tag,
            reasoning_template=reasoning_template,
        )
        self.allow_explanation_between_tags = allow_explanation_between_tags

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

        # When reasoning_template=True and target is think tag,
        # the opening </think> tag is omitted by the chat template.
        # We check for </think> closing tag and content before it.
        if self.reasoning_template and self.target_tag == self.think_tag:
            think_close = f"</{self.think_tag}>"
            close_count = task.model_output.count(think_close)
            if close_count > 1:
                return RewardFunctionScore(
                    score=0.0,
                    info=f"Multiple </{self.think_tag}> tags found.",
                )
            if close_count == 0:
                return RewardFunctionScore(
                    score=0.0,
                    info=f"No </{self.think_tag}> closing tag found.",
                )
            # Content is everything before </think>
            close_index = task.model_output.find(think_close)
            raw_content = task.model_output[:close_index].strip()
            # Check there's actual reasoning content
            if not raw_content:
                return RewardFunctionScore(
                    score=0.0,
                    info=f"Empty reasoning content before </{self.think_tag}>.",
                )
            # Check for actual nested answer tags (not quoted examples)
            # Skip instruction echoes like "<answer>[Final numeric result]</answer>" or "<answer>...</answer>"
            answer_open = f"<{self.answer_tag}>"
            answer_close = f"</{self.answer_tag}>"
            if answer_open in raw_content or answer_close in raw_content:
                import re as re_module
                # Strip all placeholder patterns: [anything], ..., answer here, your answer, etc.
                content_cleaned = raw_content
                # Remove <answer>[...]</answer>
                content_cleaned = re_module.sub(
                    rf"<{self.answer_tag}>\[.*?\]</{self.answer_tag}>", "", content_cleaned, flags=re_module.DOTALL
                )
                # Remove <answer>...</answer>
                content_cleaned = re_module.sub(
                    rf"<{self.answer_tag}>\.\.\.</{self.answer_tag}>", "", content_cleaned, flags=re_module.DOTALL
                )
                # Check remaining content for real tags
                has_real_answer = False
                if answer_open in content_cleaned or answer_close in content_cleaned:
                    # Check if any remaining <answer>...</answer> has real content
                    answer_pattern = f"<{self.answer_tag}>(.*?)</{self.answer_tag}>"
                    matches = re_module.findall(answer_pattern, content_cleaned, re_module.DOTALL)
                    for match in matches:
                        stripped = match.strip()
                        if stripped.lower() in ("answer here", "your answer", "final answer", "the answer"):
                            continue
                        has_real_answer = True
                        break
                    if not has_real_answer and answer_close in content_cleaned:
                        has_real_answer = True
                if has_real_answer:
                    return RewardFunctionScore(
                        score=0.0,
                        info=f"<{self.answer_tag}> tag found inside reasoning section. Tags cannot be nested.",
                    )
            return RewardFunctionScore(score=1.0, info=f"Valid {self.target_tag} format (reasoning template).")

        # Note: Do NOT use re.escape() on tag_start/end since they are literal tags, not regex patterns
        pattern = f"{tag_start}(.*?){tag_end}"
        matches = re.findall(pattern, task.model_output or "", re.DOTALL)

        # count how many tag_start and tag_end are present
        # But for answer_tag, exclude tags that are inside the reasoning section (before </think>)
        # since those are typically prompt echoes like "Response Structure: <answer>[...]</answer>"
        if self.target_tag == self.answer_tag and self.reasoning_template:
            think_close = f"</{self.think_tag}>"
            close_idx = task.model_output.find(think_close)
            if close_idx >= 0:
                # Only count tags AFTER the reasoning section
                content_after_reasoning = task.model_output[close_idx + len(think_close):]
                start_count = content_after_reasoning.count(tag_start)
                end_count = content_after_reasoning.count(tag_end)
            else:
                start_count = task.model_output.count(tag_start)
                end_count = task.model_output.count(tag_end)
        else:
            start_count = task.model_output.count(tag_start)
            end_count = task.model_output.count(tag_end)

        if start_count > 1 or end_count > 1:
            return RewardFunctionScore(
                score=0.0,
                info=f"Multiple <{self.target_tag}> tags found.",
            )

        if not matches:
            return RewardFunctionScore(
                score=0.0,
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
                    score=0.0,
                    info=f"Content found before <{self.target_tag}> opening tag. Tags must appear at the start.",
                )
        elif self.target_tag == self.answer_tag:
            # Answer tag can have think tag before it, but check for other content
            # Remove the think tag section if present
            if content_before_tag:
                if self.reasoning_template:
                    # With reasoning template, the "think section" is everything up to and
                    # including </think> (which marks the end of auto-injected reasoning)
                    think_close = f"</{self.think_tag}>"
                    close_idx = content_before_tag.find(think_close)
                    if close_idx >= 0:
                        # Everything up to </think> is the reasoning section; ignore it
                        content_without_think = content_before_tag[close_idx + len(think_close):].strip()
                    else:
                        content_without_think = content_before_tag.strip()
                else:
                    think_tag_pattern = f"<{self.think_tag}>.*?</{self.think_tag}>"
                    content_without_think = re.sub(
                        think_tag_pattern, "", content_before_tag, flags=re.DOTALL
                    ).strip()
                if content_without_think and not self.allow_explanation_between_tags:
                    return RewardFunctionScore(
                        score=0.0,
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
                    score=0.0,
                    info=f"<{forbidden_tag}> tag found inside <{self.target_tag}> tag. Tags cannot be nested.",
                )

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
            return RewardFunctionScore(score=1.0, info="Valid tag with content.")
        else:
            return RewardFunctionScore(score=0.0, info="Empty answer tag.")
