from .reward_function import RewardFunctionScore
from .math import MathRewardFunction
from .puzzle import PuzzleRewardFunction
from .lang_consistency import LangConsistencyRewardFunction
from .reasoning_steps import ReasoningStepsRewardFunction
from .format import FormatRewardFunction
from .llm_judge import LLMJudgeRewardFunction
from .length import LengthRewardFunction


def get_reward_functions(
    timeout: int = 10, answer_tag: str = "answer", think_tag: str = "think"
):
    """Get task-type specific reward functions (math, puzzle).

    Note: Auxiliary reward functions (lang_consistency, reasoning_steps, etc.)
    are initialized separately via CurriculumLearning._initialize_aux_reward_functions()
    and should not be included here.

    Args:
        timeout: Timeout for reward function execution
        answer_tag: Tag used to extract answers from model responses
        think_tag: Tag used to extract reasoning from model responses

    Returns:
        Dictionary mapping task types to their reward functions
    """
    reward_fns = {
        "math": MathRewardFunction(
            "math", timeout=timeout, answer_tag=answer_tag, think_tag=think_tag
        ),
        "puzzle": PuzzleRewardFunction(
            "puzzle", timeout=timeout, answer_tag=answer_tag, think_tag=think_tag
        ),
    }

    return reward_fns
