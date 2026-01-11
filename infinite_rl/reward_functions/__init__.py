from .math import MathRewardFunction
from .summarization import SummarizationRewardFunction
from .coding import CodingRewardFunction
from .html import HtmlRewardFunction


def get_reward_functions():
    reward_fns = {
        "math": MathRewardFunction("math"),
        "summarization": SummarizationRewardFunction("summarization"),
        "coding": CodingRewardFunction("coding"),
        "html": HtmlRewardFunction("html"),
    }

    return reward_fns
