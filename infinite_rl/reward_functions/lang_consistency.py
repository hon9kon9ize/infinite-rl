from typing import Union
from .reward_function import RewardFunction, RewardFunctionScore
from ..utils.parser_utils import extract_answer_tags
import pycld2 as cld2

# cantonesedetect is an optional specialized detector included in requirements
try:
    from cantonesedetect import CantoneseDetector
except Exception:
    CantoneseDetector = None


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

    def initialize(self):
        self.initialized = True

    def _yue_ratio(self, text: str) -> float:
        """Return a ratio (0..1) representing how Cantonese the text appears.

        Falls back to 0.0 if detector is unavailable or no segments.
        """
        if CantoneseDetector is None:
            return 0.0

        # Use cld2 to pre-check if the text is Chinese; if not, it's unlikely to be Cantonese
        try:
            is_reliable, _, details = cld2.detect(text.encode("utf-8"))
            if is_reliable:
                lang = details[0][1]
                if lang not in ["zh-Hant", "zh"]:
                    return 0.0
        except Exception:
            pass

        detector = CantoneseDetector(split_seg=True, get_analysis=True)
        judgement, document_features = detector.judge(text)
        total = len(document_features.document_segments_features)
        if total == 0:
            return 0.0
        canto_segments = sum(
            1
            for seg in document_features.document_segments_features
            if seg.canto_feature > seg.swc_feature
        )
        return canto_segments / total

    def _detect_lang(self, text: str) -> Union[str, None]:
        try:
            _, _, details = cld2.detect(text.encode("utf-8"))
            if details:
                return details[0][1]
        except Exception:
            pass
        return None

    def _detect_lang_details(self, text: str):
        """Return CLD2 details list: [(language_name, language_code, byte_count), ...]"""
        try:
            _, _, details = cld2.detect(text.encode("utf-8"))
            return details or []
        except Exception:
            return []

    def compute_reward(
        self,
        model_output: str,
        expected_output: Union[str, int, None],
    ) -> RewardFunctionScore:
        # Ensure initialized
        if not self.initialized:
            self.initialize()

        # 1. Format: require <answer> tags
        matches = extract_answer_tags(model_output, tag=self.answer_tag)
        if not matches:
            content = model_output.strip()
        else:
            # remove answer tag content from content (single-string result)
            content = model_output.replace(matches, "").strip()

        # Determine expected language code or dialect
        # expected_output is expected to be a short language target like 'en', 'zh', or 'yue'.
        expected = None
        if isinstance(expected_output, str) and expected_output.strip():
            norm = expected_output.strip().lower()
            # Accept common aliases
            if norm in ("en", "zh", "yue", "zh-hant"):
                expected = norm
            else:
                # Not a recognized language code; leave as None so we fall back to detection later
                expected = None

        if not expected:
            # As a last resort, attempt to detect the language of the prompt/answer content
            detected_from_answer = self._detect_lang(content)
            if detected_from_answer:
                # We have something but without a stated expectation: treat as auxiliary signal
                return RewardFunctionScore(score=1.0)

            return RewardFunctionScore(
                score=0.0,
                error_msg={
                    "lang_consistency": "Could not determine expected language from the Answer or example."
                },
            )

        # Graded mapping to handle overlap between Chinese variants and Cantonese
        mapping = {
            "zh": {"zh": 1.0, "zh-hant": 0.25, "yue": 0.25, "en": 0.0},
            "zh-hant": {"zh": 0.25, "zh-hant": 1.0, "yue": 0.25, "en": 0.0},
            "yue": {"zh": 0.25, "zh-hant": 0.5, "yue": 1.0, "en": 0.0},
            "en": {"zh": 0.0, "zh-hant": 0.0, "yue": 0.0, "en": 1.0},
        }

        norm_expected = expected.lower()

        # Get detected language (CLD2) details
        details = self._detect_lang_details(content)
        norm_detected = details[0][1].lower() if details else None

        # Cantonese ratio (0..1) from specialized detector (if available)
        y_ratio = self._yue_ratio(content)

        # Use detailed CLD2 information (byte counts) to compute a weighted mapping score
        details = self._detect_lang_details(content)

        # If the expected is Cantonese, compute a weighted mapping score and combine with y_ratio
        if norm_expected == "yue":
            weighted = 0.0
            total_bytes = sum(entry[-1] for entry in details) if details else 0
            if total_bytes > 0:
                for entry in details:
                    code = entry[1]
                    b = entry[-1]
                    c = code.lower()
                    if c.startswith("zh-hant"):
                        norm = "zh-hant"
                    elif c.startswith("zh"):
                        norm = "zh"
                    else:
                        norm = c
                    weighted += (b / total_bytes) * mapping["yue"].get(norm, 0.0)

            final_score = max(y_ratio, weighted)
            return RewardFunctionScore(
                score=float(final_score),
                error_msg={
                    "lang_consistency": (
                        ("Response appears to be Cantonese.")
                        if final_score > 0
                        else "Response does not appear to be Cantonese."
                    )
                },
            )

        # For other expected languages, compute weighted score from CLD2 details
        if norm_expected in mapping:
            weighted = 0.0
            total_bytes = sum(entry[-1] for entry in details) if details else 0
            if total_bytes > 0:
                for entry in details:
                    code = entry[1]
                    b = entry[-1]
                    c = code.lower()
                    if c.startswith("zh-hant"):
                        norm = "zh-hant"
                    elif c.startswith("zh"):
                        norm = "zh"
                    else:
                        norm = c
                    weighted += (b / total_bytes) * mapping[norm_expected].get(
                        norm, 0.0
                    )

            # Consider Cantonese detector signal as potential additional evidence
            if y_ratio > 0:
                weighted = max(
                    weighted, mapping[norm_expected].get("yue", 0.0) * y_ratio
                )

            # Auxiliary signal: expose weighted mapping in score
            return RewardFunctionScore(score=float(weighted))

        # Fallback: previous prefix matching logic (binary)
        if not norm_detected:
            return RewardFunctionScore(
                score=0.0,
                error_msg={
                    "lang_consistency": "Failed to detect language of the response."
                },
            )

        if (
            norm_expected == norm_detected
            or norm_detected.startswith(norm_expected)
            or norm_expected.startswith(norm_detected)
        ):
            return RewardFunctionScore(score=1.0)

        return RewardFunctionScore(
            score=0.0,
            error_msg=f"Detected language '{norm_detected}' does not match expected '{expected}'.",
        )
