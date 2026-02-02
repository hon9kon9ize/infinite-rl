"""Whitespace Collapse Detector - Prevents reward hacking via space removal.

This reward function detects when models remove spaces to game reasoning_steps rewards
and compress tokens to fit within length limits. It analyzes the <think> tag content
and penalizes suspicious space-to-character ratios that deviate from natural language.
"""

import re
from typing import TYPE_CHECKING
from .reward_function import RewardFunction, RewardFunctionScore

if TYPE_CHECKING:
    from ..task import Task


class WhitespaceCollapseRewardFunction(RewardFunction):
    """Detects and penalizes whitespace collapse (reward hacking via space removal).

    This function identifies when models remove spaces from reasoning text to:
    1. Maximize reasoning_steps rewards (more words fit in fewer tokens)
    2. Fit within length penalties (token-count optimization)

    Only applies to English language reasoning (checks task.language == 'en' or similar).
    Only analyzes the <think> tag content, not the answer.

    Detection Logic:
    - Extracts <think> tag content for analysis
    - Calculates space-to-character ratio in the thinking text
    - Flags text >100 chars with space_ratio <5% as collapsed (natural English ~17-20% spaces)
    - Returns penalty -0.5 if detected

    Args:
        task_name: Name of this reward function (default: "whitespace_collapse")
        reasoning_language: ISO language code for reasoning to check (default: "en")
        min_text_length: Minimum chars in think tag to check (default: 100)
        space_ratio_threshold: Max space ratio before flagging (default: 0.05 = 5%)
        **kwargs: Additional RewardFunction parameters (timeout, answer_tag, think_tag, etc.)
    """

    def __init__(
        self,
        task_name: str = "whitespace_collapse",
        reasoning_language: str = "en",
        min_text_length: int = 100,
        space_ratio_threshold: float = 0.05,
        **kwargs,
    ):
        super().__init__(task_name, **kwargs)
        self.reasoning_language = reasoning_language
        self.min_text_length = min_text_length
        self.space_ratio_threshold = space_ratio_threshold

    def initialize(self):
        self.initialized = True

    def compute_reward(
        self,
        task: "Task",
        **kwargs,
    ) -> RewardFunctionScore:
        """Compute whitespace collapse penalty for task.

        Args:
            task: Task object containing model_output, language, and other metadata
            **kwargs: Additional arguments (ignored)

        Returns:
            RewardFunctionScore with:
            - score: 0.0 (no penalty) or -0.5 (collapse detected)
            - info: Description of finding
        """
        if not self.initialized:
            self.initialize()

        # Check task reasoning language (separate from programming language)
        # For puzzles: task.language is code language (javascript/python), task.reasoning_language is <think> language
        # For math: task.reasoning_language is the reasoning language
        task_reasoning_language = getattr(
            task, "reasoning_language", getattr(task, "language", "en")
        )
        # Only check for English reasoning
        if task_reasoning_language != "en":
            # Skip check for non-English reasoning
            return RewardFunctionScore(
                score=0.0,
                reward_function_name=self.task_name,
                info=f"Skipped (task reasoning_language={task_reasoning_language}, only checking English)",
            )

        model_output = task.model_output or ""
        if not model_output:
            return RewardFunctionScore(
                score=0.0,
                reward_function_name=self.task_name,
                info="No output to analyze",
            )

        # Extract thinking content from <think> tags
        think_content = self._extract_think_content(model_output)
        if not think_content:
            return RewardFunctionScore(
                score=0.0,
                reward_function_name=self.task_name,
                info="No <think> tag found",
            )

        # Analyze space ratio in thinking
        is_collapsed, penalty, info = self._analyze_whitespace(think_content)

        return RewardFunctionScore(
            score=penalty,
            reward_function_name=self.task_name,
            info=info,
        )

    def _extract_think_content(self, model_output: str) -> str:
        """Extract content between <think> tags.

        Args:
            model_output: Raw model response

        Returns:
            Stripped think tag content, or empty string if not found
        """
        if not self.think_tag:
            return ""

        pattern = f"<{self.think_tag}>(.*?)</{self.think_tag}>"
        match = re.search(pattern, model_output, re.DOTALL)

        if not match:
            return ""

        return match.group(1).strip()

    def _analyze_whitespace(self, think_content: str) -> tuple[bool, float, str]:
        """Analyze space-to-character ratio in think content.

        Args:
            think_content: Text extracted from <think> tags

        Returns:
            Tuple of (is_collapsed, penalty, info_message):
            - is_collapsed: True if whitespace collapse detected
            - penalty: 0.0 or -0.5
            - info_message: Description of finding
        """
        text_len = len(think_content)

        # Only analyze substantial reasoning (avoid false positives on short text)
        if text_len < self.min_text_length:
            return (
                False,
                0.0,
                f"Text too short ({text_len} < {self.min_text_length} chars)",
            )

        # Calculate space ratio
        space_count = think_content.count(" ")
        space_ratio = space_count / text_len if text_len > 0 else 0.0

        # Normal English has ~17-20% spaces; <5% is suspicious "compression"
        if space_ratio < self.space_ratio_threshold:
            penalty = -0.5
            info = f"Whitespace collapse detected: {space_ratio:.1%} spaces in {text_len} chars (threshold: {self.space_ratio_threshold:.1%})"
            return True, penalty, info

        # No collapse detected
        return (
            False,
            0.0,
            f"Normal spacing: {space_ratio:.1%} spaces (threshold: {self.space_ratio_threshold:.1%})",
        )
