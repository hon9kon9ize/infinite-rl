"""Repetition-based reward utilities.

Detects n-gram repetitions and returns a penalty (negative float).
"""

from collections import Counter
import re
from .reward_function import RewardFunction, RewardFunctionScore


def _tokenize(text: str) -> list:
    """
    Tokenize text into a list of tokens.

    - For Latin script (ASCII letters/digits) we return word tokens.
    - For CJK (Chinese) characters we split into single-character tokens so
      n-gram repetition detection can operate at character granularity.

    This avoids adding heavy dependencies like jieba while providing
    deterministic behavior for unit tests and typical use-cases.
    """
    text = text.lower()

    # Regex alternation: match single CJK characters (including supplementary planes and variation selectors)
    # or sequences of ASCII letters/digits. The CJK expression below covers common unified ideograph ranges
    # and recommended additional ranges/variation selectors for better coverage.
    cjk = (
        r"[\u4e00-\u9fff\u3400-\u4dbf\U00020000-\U0002a6df\U0002a700-\U0002ebef\U00030000-\U000323af"
        r"\ufa0e\ufa0f\ufa11\ufa13\ufa14\ufa1f\ufa21\ufa23\ufa24\ufa27\ufa28\ufa29\u3006\u3007]"
        r"[\ufe00-\ufe0f\U000e0100-\U000e01ef]?"
    )
    # Match either a single CJK character (with optional variation selector) or an ASCII word/token
    pattern = re.compile(rf"{cjk}|[A-Za-z0-9]+")

    tokens = pattern.findall(text)
    return [t for t in tokens if t]


def ngram_repetition_reward(text: str, n: int = 3, weight: float = -0.1) -> float:
    """Compute an n-gram repetition penalty.

    Returns a penalty (<= 0). The penalty is computed as:
      penalty = (duplicates / total_ngrams) * weight
    where duplicates = sum(count - 1 for count in counts.values() if count > 1).

    Args:
        text: model output string.
        n: size of n-gram (default: 3).
        weight: negative multiplier applied to normalized duplicate ratio (default: -0.1).

    Returns:
        float penalty (<= 0). If there are no n-grams or no duplicates, returns 0.0.
    """
    if n <= 0:
        raise ValueError("n must be > 0")

    tokens = _tokenize(text)
    if len(tokens) < n:
        return 0.0

    ngrams = [tuple(tokens[i : i + n]) for i in range(len(tokens) - n + 1)]
    counts = Counter(ngrams)
    duplicates = sum(count - 1 for count in counts.values() if count > 1)

    if not ngrams:
        return 0.0

    penalty = (duplicates / len(ngrams)) * weight
    return float(penalty)


class RepetitionRewardFunction(RewardFunction):
    """Reward function that applies an n-gram repetition penalty as a lightweight reward.

    The correctness_score is computed as max(0.0, 1.0 + penalty) where penalty <= 0.
    """

    def __init__(
        self,
        task_name: str = "repetition",
        timeout: int = 5,
        n: int = 3,
        weight: float = -0.1,
        answer_tag: str = "answer",
        think_tag: str = "think",
    ):
        super().__init__(
            task_name, timeout=timeout, answer_tag=answer_tag, think_tag=think_tag
        )
        self.n = n
        self.weight = weight

    def initialize(self):
        self.initialized = True

    def compute_reward(
        self,
        model_output: str,
        **kwargs,
    ) -> RewardFunctionScore:
        if not self.initialized:
            self.initialize()

        if not model_output:
            return RewardFunctionScore(score=0.0)

        text = self.extract_tag(model_output, **kwargs)

        if not text:
            return RewardFunctionScore(
                score=0.0,
                error_msg={"repetition": "No content found in the specified tag."},
            )

        penalty = ngram_repetition_reward(text, n=self.n, weight=self.weight)
        correctness = max(0.0, 1.0 + float(penalty))

        return RewardFunctionScore(score=float(correctness))
