from .math import MathRewardFunction
from .coding import CodingRewardFunction
from .language import LanguageRewardFunction


def get_reward_functions(timeout: int = 10):
    reward_fns = {
        "math": MathRewardFunction("math", timeout=timeout),
        "coding": CodingRewardFunction("coding", timeout=timeout),
        "language": LanguageRewardFunction("language", timeout=timeout),
    }

    return reward_fns
