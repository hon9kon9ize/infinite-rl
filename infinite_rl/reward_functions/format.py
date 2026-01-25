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
        expected_output: Union[str, int, None] = None,
        target_tag: str = None,
    ) -> RewardFunctionScore:
        if not self.initialized:
            self.initialize()

        target_tag = target_tag if target_tag is not None else self.answer_tag

        matches = extract_tag(model_output, tag=target_tag)
        if not matches:
            return RewardFunctionScore(
                score=0.0, error_msg={"format": f"Missing <{target_tag}> tags"}
            )

        content = matches.strip()

        # For tasks that encourage explicit reasoning, ensure a <think> block exists
        if self.task_name == "reasoning_steps":
            think_content = extract_tag(model_output, tag=self.think_tag)
            if not think_content:
                return RewardFunctionScore(
                    score=0.0,
                    error_msg={
                        "format": f"Missing <{self.think_tag}> tags in response for reasoning task."
                    },
                )

        # Code-like tasks
        if self.task_name in ("python", "javascript", "coding"):
            # Proper fenced block with language
            pattern_lang = rf"```(?:{self.task_name})\b.*?```"
            if re.search(pattern_lang, content, re.IGNORECASE | re.DOTALL):
                return RewardFunctionScore(score=1.0)

            # Any fenced block (closed)
            if re.search(r"```.*?```", content, re.DOTALL):
                # fenced but no language specified
                return RewardFunctionScore(score=0.5)

            # Malformed fenced block (unclosed backticks)
            if "```" in content:
                return RewardFunctionScore(
                    score=0.0, error_msg={"format": "Malformed code block"}
                )

            # No fenced block but some code-like content
            if any(
                kw in content
                for kw in ("def ", "class ", "import ", "console.log", "print(")
            ):
                return RewardFunctionScore(score=0.5)

            # If <answer> tag present with non-empty content but no fenced block, give partial credit (0.5)
            if content.strip():
                return RewardFunctionScore(score=0.5)

            return RewardFunctionScore(
                score=0.0,
                error_msg={"format": f"No code block found inside <{target_tag}>"},
            )

        # Math tasks: expect a simple numeric value inside the tag (no backticks)
        if self.task_name == "math":
            # Disallow code fences for math answers
            if "```" in content:
                return RewardFunctionScore(
                    score=0.0,
                    error_msg={"format": f"Unexpected code fence in <{target_tag}>"},
                )

            if content.strip():
                return RewardFunctionScore(score=1.0)
            return RewardFunctionScore(
                score=0.0, error_msg={"format": f"Empty <{target_tag}> tag"}
            )

        # Generic fallback
        return RewardFunctionScore(score=1.0 if content else 0.0)
