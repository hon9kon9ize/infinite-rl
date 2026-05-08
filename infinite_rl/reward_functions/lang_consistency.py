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

    def _is_cantonese_without_cld2(self, text: str) -> bool:
        """Check Cantonese directly without CLD2 source-language prefiltering."""
        judgement = self.yue_detector(text)
        if judgement in ["cantonese", "mixed"]:
            return True
        if judgement == "neutral":
            # `neutral` is useful for short Cantonese snippets such as common
            # phrases, but English/French text can also be neutral. Require at
            # least some CJK signal when accepting neutral as Cantonese-like.
            return any("\u4e00" <= char <= "\u9fff" for char in text)
        return False

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
                return output[start + len(open_tag):end].strip()
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

        # Get expected language. Pre-reasoning trains CoT language only; the
        # final response can be multilingual, so ignore task.language there.
        if task.task_type == "pre_reasoning":
            expected_output = task.reasoning_language or kwargs.get(
                "target_language", self.target_language
            )
        elif task.task_type == "truthy":
            expected_output = kwargs.get("target_language", self.target_language)
            if task.language:
                expected_output = task.language
        else:
            # For math/puzzle tasks, use reasoning_language (the CoT language)
            expected_output = task.reasoning_language or self.target_language

        # Determine which content to check based on task type
        if task.task_type == "pre_reasoning":
            # For pre-reasoning, only check the CoT language. The content after
            # </think> may legitimately be any language in multilingual data.
            content = self._extract_reasoning_content(task.model_output or "")
        elif task.task_type == "truthy":
            # For legacy truthy tasks, check content outside <think> tags.
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

        # Cantonese detection should not be gated by CLD2. Translation and
        # multilingual prompts often quote source text in French/English/etc.,
        # which can cause CLD2 to report the source language even when the CoT
        # itself is Cantonese.
        if norm_expected == "yue":
            is_cantonese = self._is_cantonese_without_cld2(content)
            if is_cantonese:
                return RewardFunctionScore(score=1.0, info="")
            return RewardFunctionScore(
                score=0.0,
                info="Expected Cantonese but content was not classified as Cantonese.",
            )

        # Get detected language (CLD2) details
        _, _, details = cld2.detect(content.encode("utf-8"))
        norm_detected = details[0][1].lower() if details else None

        if not norm_detected:
            return RewardFunctionScore(
                score=0.0,
                info="Failed to detect language of the content.",
            )

        if norm_expected == norm_detected:
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
