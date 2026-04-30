import re
from typing import Union, TYPE_CHECKING
from .reward_function import RewardFunction, RewardFunctionScore

if TYPE_CHECKING:
    from ..task import Task

# English reasoning step indicators
_EN_INDICATORS = [
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

# Cantonese / Chinese reasoning step indicators
_CN_INDICATORS = [
    # Numbered steps
    "第一步",
    "第二步",
    "第三步",
    "第四步",
    "第五步",
    "第一",
    "第二",
    "第三",
    "第四",
    "第五",
    # Ordering / sequencing
    "首先",
    "其次",
    "然後",
    "之後",
    "接著",
    "最後",
    # Cantonese-specific sequencing
    "跟住",
    "然後",
    "之後",
    # Logical connectors
    "所以",
    "因此",
    "因為",
    "由於",
    "總結",
    "結論",
    # Analysis / thinking patterns
    "分析",
    "考慮",
    "假設",
    "如果",
    "但係",
    "另一方面",
    # Cantonese thinking phrases
    "我要",
    "我先",
    "我想",
    "我哋",
    "呢個",
    "等我",
    "等等",
    # Calculation / verification
    "計算",
    "驗證",
    "確認",
    "諗下",
    "再諗",
    "諗諗",
    "檢查",
    "check",
    "睇下",
    "睇返",
]


class ReasoningStepsRewardFunction(RewardFunction):
    """Reward function that gives bonuses based on reasoning step indicators.

    The function looks for reasoning content inside <think> tags and checks
    for presence of step/analysis indicators. Supports both English and
    Cantonese/Chinese reasoning.

    Language selection is based on task.reasoning_language:
    - 'yue' or 'zh' -> Chinese/Cantonese indicators
    - 'en' or default -> English indicators

    Returns:
    - 1.0: Three or more indicators found (strong reasoning)
    - 0.7: Two indicators found (good reasoning)
    - 0.5: One indicator found (basic reasoning)
    - 0.0: No indicators found (neutral, no penalty)
    """

    def __init__(
        self,
        task_name: str = "reasoning_steps",
        timeout: int = 5,
        target_tag: str = "think",
        reasoning_template: bool = False,
        **kwargs,
    ):
        super().__init__(
            task_name,
            timeout=timeout,
            target_tag=target_tag,
            reasoning_template=reasoning_template,
            **kwargs,
        )

    def initialize(self):
        self.initialized = True

    def _get_indicators(self, reasoning_language: str) -> list:
        """Select indicator list based on reasoning language."""
        lang = (reasoning_language or "en").lower()
        if lang in ("yue", "zh", "zh-hant", "zh-hans"):
            return _CN_INDICATORS
        return _EN_INDICATORS

    def _extract_thinking_content(self, model_output: str) -> str:
        """Extract reasoning content from model output.

        Handles both tag formats:
        - Standard <think>...</think> (XML comment style)
        - <think>...</think> (reasoning template: no opening tag)
        """
        output = model_output or ""

        # Try standard <think>...</think> format first
        if "</think>" in output and "</think>" in output:
            start = output.index("</think>")
            end = output.index("</think>", start)
            if start < end:
                return output[start:end].strip()

        # Fallback: reasoning_template mode — everything before </think>
        if "</think>" in output:
            return output.split("</think>")[0].strip()

        return ""

    def compute_reward(
        self,
        task: "Task",
        **kwargs,
    ) -> RewardFunctionScore:
        if not self.initialized:
            self.initialize()

        if not task.model_output:
            return RewardFunctionScore(score=0.0)

        think_content = self._extract_thinking_content(task.model_output)
        if not think_content:
            return RewardFunctionScore(
                score=0.0,
                info="No reasoning content found.",
            )

        indicators = self._get_indicators(task.reasoning_language)
        thinking_lower = think_content.lower()

        found_count = sum(1 for word in indicators if word in thinking_lower)

        if found_count >= 3:
            bonus = 1.0
        elif found_count >= 2:
            bonus = 0.7
        elif found_count > 0:
            bonus = 0.5
        else:
            # No penalty — having reasoning content is validated by format_think
            bonus = 0.0

        return RewardFunctionScore(
            score=float(bonus),
            info=f"Found {found_count} reasoning indicators.",
        )
