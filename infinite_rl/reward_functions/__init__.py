from .math import MathRewardFunction
from .summarization import SummarizationRewardFunction
from .coding import CodingRewardFunction
from .html import HtmlRewardFunction


def get_reward_functions(timeout: int = 5):
    reward_fns = {
        "math": MathRewardFunction("math", timeout=timeout),
        "summarization": SummarizationRewardFunction("summarization", timeout=timeout),
        "coding": CodingRewardFunction("coding", timeout=timeout),
        "html": HtmlRewardFunction("html", timeout=timeout),
    }

    return reward_fns
