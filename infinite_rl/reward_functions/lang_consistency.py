from typing import Union
from collections import defaultdict
import pycld2 as cld2
from cantonesedetect import CantoneseDetector
from .reward_function import RewardFunction, RewardFunctionScore


class LangConsistencyRewardFunction(RewardFunction):
    """Reward function that checks language/dialect consistency of model responses.

    expected_output: a language code (e.g., 'en', 'zh', 'zh-Hant') or a dialect string 'yue' (Cantonese).
    If an example sentence is provided in the Answer, the function will try to infer the expected language
    from that text as a fallback.
    """

    def __init__(
        self,
        task_name: str = "lang_consistency",
        timeout: int = 5,
        answer_tag: str = "answer",
        think_tag: str = "think",
    ):
        super().__init__(
            task_name, timeout=timeout, answer_tag=answer_tag, think_tag=think_tag
        )
        self.yue_detector = CantoneseDetector(split_seg=True, get_analysis=True)

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
        model_output: str,
        expected_output: Union[str, int, float, None],
    ) -> RewardFunctionScore:
        # Ensure initialized
        if not self.initialized:
            self.initialize()

        content = model_output.strip()
        norm_expected = expected_output.lower().strip()

        # Determine expected language code or dialect
        # expected_output is expected to be a short language target like 'en', 'zh', or 'yue'.
        if norm_expected not in ["en", "zh", "yue", "zh-hant"]:
            return RewardFunctionScore(
                score=0.0,
                error_msg={
                    "lang_consistency": "Expected output must be a language code like 'en', 'zh', 'zh-Hant', or 'yue'."
                },
            )

        # Get detected language (CLD2) details
        _, _, details = cld2.detect(content.encode("utf-8"))
        norm_detected = details[0][1].lower() if details else None

        if not norm_detected:
            return RewardFunctionScore(
                score=0.0,
                error_msg={
                    "lang_consistency": "Failed to detect language of the response."
                },
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
        print(lang_ratio[norm_expected], lang_ratio[norm_detected])

        final_score = lang_ratio[norm_expected]

        return RewardFunctionScore(
            score=final_score,
            error_msg=(
                {
                    "lang_consistency": f"Detected language '{norm_detected}' does not match expected '{norm_expected}'."
                }
                if norm_expected != norm_detected
                else None
            ),
        )
