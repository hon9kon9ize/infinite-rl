from .math import MathRewardFunction
from .coding import CodingRewardFunction
from .lang_consistency import LangConsistencyRewardFunction
from .reasoning_steps import ReasoningStepsRewardFunction


def get_reward_functions(timeout: int = 10):
    reward_fns = {
        "math": MathRewardFunction("math", timeout=timeout),
        "coding": CodingRewardFunction("coding", timeout=timeout),
        # New standardized name
        "lang_consistency": LangConsistencyRewardFunction(
            "lang_consistency", timeout=timeout
        ),
        "reasoning_steps": ReasoningStepsRewardFunction(
            "reasoning_steps", timeout=timeout
        ),
    }

    return reward_fns
