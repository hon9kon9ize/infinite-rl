from .math import MathRewardFunction
from .coding import PythonRewardFunction, JavascriptRewardFunction
from .lang_consistency import LangConsistencyRewardFunction
from .reasoning_steps import ReasoningStepsRewardFunction
from .repetition import RepetitionRewardFunction
from .length import LengthRewardFunction


def get_reward_functions(timeout: int = 10):
    reward_fns = {
        "math": MathRewardFunction("math", timeout=timeout),
        "python": PythonRewardFunction("python", timeout=timeout),
        "javascript": JavascriptRewardFunction("javascript", timeout=timeout),
        # New standardized name
        "lang_consistency": LangConsistencyRewardFunction(
            "lang_consistency", timeout=timeout
        ),
        "reasoning_steps": ReasoningStepsRewardFunction(
            "reasoning_steps", timeout=timeout
        ),
        "repetition": RepetitionRewardFunction("repetition", timeout=timeout),
        "length": LengthRewardFunction("length", timeout=timeout),
    }

    return reward_fns
