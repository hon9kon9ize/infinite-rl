from typing import Union, TYPE_CHECKING
from collections import defaultdict
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
        answer_tag_excluded=True,
        **kwargs,
    ):
        from cantonesedetect import CantoneseDetector

        super().__init__(task_name, **kwargs)
        self.yue_detector = CantoneseDetector(split_seg=True, get_analysis=True)
        self.answer_tag_excluded = answer_tag_excluded

    def initialize(self):
        self.initialized = True

    def _yue_ratio(self, text: str) -> float:
        """Return a ratio (0..1) representing how Cantonese the text appears."""
        judgement, _ = self.yue_detector.judge(text)

        if judgement in ["cantonese", "neutral"]:
            return 1.0
        elif judgement == "swc":
            return 0.25
        elif judgement in ["mixed", "cantonese_quotes_in_swc", "mixed_quotes_in_swc"]:
            return 0.5

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

        # Extract content using answer_tag_excluded parameter
        content = extract_tag_util(
            task.model_output or "",
            tag=self.target_tag,
            exclude=self.answer_tag_excluded,
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
                score=0.0,
                info=f"Failed to detect language of the response inside <{self.target_tag}>.",
            )

        # Cantonese ratio (0..1) from specialized detector (if available)
        y_ratio = self._yue_ratio(content)
        norm_detected = (
            "yue" if y_ratio == 1.0 and norm_expected == "yue" else norm_detected
        )
        lang_ratio = defaultdict(float)
        total_bytes = sum(entry[-1] for entry in details) if details else 0
        if total_bytes > 0:
            for entry in details:
                code = entry[1]
                b = entry[-1]
                c = code.lower()
                lang_ratio[c] += b / total_bytes
        lang_ratio["yue"] = max(y_ratio, lang_ratio["yue"])
        final_score = lang_ratio[norm_expected]

        return RewardFunctionScore(
            score=final_score,
            info=(
                f"Detected language '{norm_detected}' does not match expected '{norm_expected}'."
                if norm_expected != norm_detected
                else ""
            ),
        )
