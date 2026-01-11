from typing import Tuple, Union, Callable, Any
from dataclasses import dataclass


@dataclass
class RewardFunctionScore:
    format_score: float
    correctness_score: float
    error_msg: str = ""


class RewardFunction:
    def __init__(self, task_name: str):
        self.task_name = task_name
        self.initialized = False

    def initialize(self):
        raise NotImplementedError("This method should be overridden by subclasses.")

    def compute_reward(
        self, model_output: str, expected_output: Union[str, int, Callable]
    ) -> RewardFunctionScore:
        raise NotImplementedError("This method should be overridden by subclasses.")
