from .math import MathRewardFunction
from .summarization import SummarizationRewardFunction


def get_reward_functions():
    reward_fns = {
        "math": MathRewardFunction("math"),
        "summarization": SummarizationRewardFunction("summarization"),
    }

    # Initialize the reward function
    for task_type in reward_fns:
        reward_fn = reward_fns[task_type]
        reward_fn.initialize()

    return reward_fns
