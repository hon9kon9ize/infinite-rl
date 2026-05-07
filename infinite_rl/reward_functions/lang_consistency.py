from typing import Union, TYPE_CHECKING
from .reward_function import RewardFunction, RewardFunctionScore

if TYPE_CHECKING:
    from ..task import Task


class LangConsistencyRewardFunction(RewardFunction):
    """Reward function that checks language/dialect consistency of model responses.

    expected_output: a language code (e.g., 'en', 'zh', 'zh-Hant') or a dialect string 'yue' (Cantonese).
    If an example sentence is provided in the Answer, the function will try to infer the expected language
    from that text as a fallback.
    """

    def __init__(
        self,
        task_name: str = "lang_consistency",
        tag_excluded=False,
        target_language: str = "en",
        **kwargs,
    ):
        from cantofilter import judge as yue_detector

        super().__init__(task_name, **kwargs)
        self.yue_detector = yue_detector
        self.tag_excluded = tag_excluded
        self.target_language = target_language

    def initialize(self):
        self.initialized = True

    def _is_cantonese(self, text: str) -> bool:
        """Return True if the text is Cantonese, False otherwise."""
        judgement = self.yue_detector(text)
        return judgement in ["cantonese", "mixed", "neutral"]

    def _extract_reasoning_content(self, model_output: str) -> str:
        """Extract reasoning/CoT content from model output.

        For reasoning_template=True: everything before </think> tag.
        For standard mode: content inside <think>...</think> tags.
        """
        output = model_output or ""
        open_tag = "<think>"
        close_tag = "</think>"
        # Standard mode: look for <think>...</think>
        if open_tag in output and close_tag in output:
            start = output.index(open_tag)
            end = output.index(close_tag, start)
            if start < end:
                return output[start:end].strip()
        # Reasoning template mode (closing tag only): everything before </think>
        if close_tag in output:
            return output.split(close_tag)[0].strip()
        return ""

    def _detect_language(self, content: str):
        """Detect language of content using CLD2."""
        import pycld2 as cld2
        _, _, details = cld2.detect(content.encode("utf-8"))
        return details[0][1].lower() if details else None

    def _check_cantonese(self, content: str) -> bool:
        """Check if content is Cantonese."""
        return self._is_cantonese(content)

    def compute_reward(
        self,
        task: "Task",
        **kwargs,
    ) -> RewardFunctionScore:
        import pycld2 as cld2
        from ..utils.parser_utils import extract_tag as extract_tag_util

        # Ensure initialized
        if not self.initialized:
            self.initialize()

        # Get expected language: prefer task.language for chat tasks and
        # task.reasoning_language for math/puzzle.
        if task.task_type in ("truthy", "pre_reasoning"):
            expected_output = kwargs.get("target_language", self.target_language)
            if task.language:
                expected_output = task.language
        else:
            # For math/puzzle tasks, use reasoning_language (the CoT language)
            expected_output = task.reasoning_language or self.target_language

        # Determine which content to check based on task type
        if task.task_type in ("truthy", "pre_reasoning"):
            # For chat tasks, check content outside <think> tags (the final response)
            content = extract_tag_util(
                task.model_output or "",
                tag=self.target_tag,
                exclude=self.tag_excluded,
            ).strip()
        else:
            # For math/puzzle tasks, check the reasoning/CoT content (inside <think> tags)
            content = self._extract_reasoning_content(task.model_output or "")

        if not content:
            return RewardFunctionScore(
                score=0.0,
                info=f"No content found for language check.",
            )

        norm_expected = expected_output.lower()

        # Determine expected language code or dialect
        # expected_output is expected to be a short language target like 'en', 'zh', or 'yue'.
        if norm_expected not in ["en", "zh", "yue", "zh-hant"]:
            # For programming languages (python, javascript), skip the check
            return RewardFunctionScore(
                score=1.0,
                info=f"Skipping language check for '{norm_expected}' (not a natural language target)",
            )

        # Get detected language (CLD2) details
        _, _, details = cld2.detect(content.encode("utf-8"))
        norm_detected = details[0][1].lower() if details else None

        if not norm_detected:
            return RewardFunctionScore(
                score=0.0,
                info="Failed to detect language of the content.",
            )

        # Check if Cantonese is expected and detected
        if norm_expected == "yue" and norm_detected in ["zh", "yue", "zh-hant"]:
            is_cantonese = self._is_cantonese(content)

            if is_cantonese:
                final_score = 1.0
                info_msg = ""
            else:
                final_score = 0.0
                info_msg = f"Expected Cantonese but detected '{norm_detected}' which is not Cantonese."

        elif norm_expected == norm_detected:
            # Simple binary scoring: 1.0 if match, -1.0 if mismatch
            final_score = 1.0
            info_msg = ""
        else:
            final_score = 0.0
            info_msg = f"Detected language '{norm_detected}' does not match expected '{norm_expected}'."

        return RewardFunctionScore(
            score=final_score,
            info=info_msg,
        )
