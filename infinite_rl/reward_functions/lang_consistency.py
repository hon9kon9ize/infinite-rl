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
        **kwargs,
    ):
        from cantonesedetect import CantoneseDetector

        super().__init__(task_name, **kwargs)
        self.yue_detector = CantoneseDetector(split_seg=True, get_analysis=True)
        self.tag_excluded = tag_excluded

    def initialize(self):
        self.initialized = True

    def _is_cantonese(self, text: str) -> bool:
        """Return True if the text is Cantonese, False otherwise."""
        judgement, _ = self.yue_detector.judge(text)
        return judgement in ["cantonese", "neutral"]

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

        # Get expected language from task
        expected_output = task.language

        # Extract content using tag_excluded parameter
        content = extract_tag_util(
            task.model_output or "",
            tag=self.target_tag,
            exclude=self.tag_excluded,
        ).strip()

        if not content:
            return RewardFunctionScore(
                score=0.0,
                info=f"No content found in the <{self.target_tag}> tag.",
            )

        norm_expected = expected_output.lower()

        # Determine expected language code or dialect
        # expected_output is expected to be a short language target like 'en', 'zh', or 'yue'.
        if norm_expected not in ["en", "zh", "yue", "zh-hant"]:
            return RewardFunctionScore(
                score=0.0,
                info=f"Expected output must be a language code like 'en', 'zh', 'zh-Hant', or 'yue'. Received: {norm_expected}",
            )

        # Get detected language (CLD2) details
        _, _, details = cld2.detect(content.encode("utf-8"))
        norm_detected = details[0][1].lower() if details else None

        if not norm_detected:
            return RewardFunctionScore(
                score=-1.0,
                info=f"Failed to detect language of the response inside <{self.target_tag}>.",
            )

        # Check if Cantonese is expected and detected
        if norm_expected == "yue":
            is_cantonese = self._is_cantonese(content)
            if is_cantonese:
                norm_detected = "yue"

        # Simple binary scoring: 1.0 if match, -1.0 if mismatch
        if norm_expected == norm_detected:
            final_score = 1.0
            info_msg = ""
        else:
            final_score = -1.0
            info_msg = f"Detected language '{norm_detected}' does not match expected '{norm_expected}'."

        return RewardFunctionScore(
            score=final_score,
            info=info_msg,
        )
