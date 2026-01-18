from .math import MathRewardFunction
from .coding import CodingRewardFunction
from .lang_consistency import LangConsistencyRewardFunction
from .summarization import SummarizationRewardFunction


def get_reward_functions(timeout: int = 10):
    reward_fns = {
        "math": MathRewardFunction("math", timeout=timeout),
        "coding": CodingRewardFunction("coding", timeout=timeout),
        # New standardized name
        "lang_consistency": LangConsistencyRewardFunction(
            "lang_consistency", timeout=timeout
        ),
        "summarization": SummarizationRewardFunction("summarization", timeout=timeout),
    }

    return reward_fns
