from .reward_function import RewardFunctionScore
from .math import MathRewardFunction
from .puzzle import PuzzleRewardFunction
from .lang_consistency import LangConsistencyRewardFunction
from .reasoning_steps import ReasoningStepsRewardFunction
from .repetition import RepetitionRewardFunction
from .length import LengthRewardFunction
from .format import FormatRewardFunction


def get_reward_functions(
    timeout: int = 10, answer_tag: str = "answer", think_tag: str = "think"
):
    reward_fns = {
        "math": MathRewardFunction(
            "math", timeout=timeout, answer_tag=answer_tag, think_tag=think_tag
        ),
        "puzzle": PuzzleRewardFunction(
            "puzzle", timeout=timeout, answer_tag=answer_tag, think_tag=think_tag
        ),
        # New standardized name
        "lang_consistency": LangConsistencyRewardFunction(
            "lang_consistency",
            timeout=timeout,
            answer_tag=answer_tag,
            think_tag=think_tag,
        ),
        "reasoning_steps": ReasoningStepsRewardFunction(
            "reasoning_steps",
            timeout=timeout,
            answer_tag=answer_tag,
            think_tag=think_tag,
        ),
        "repetition": RepetitionRewardFunction(
            "repetition", timeout=timeout, answer_tag=answer_tag, think_tag=think_tag
        ),
        "length": LengthRewardFunction(
            "length", timeout=timeout, answer_tag=answer_tag, think_tag=think_tag
        ),
        # Format-only reward (separate from main task correctness)
        "format": FormatRewardFunction(
            "format", timeout=timeout, answer_tag=answer_tag, think_tag=think_tag
        ),
    }

    return reward_fns
