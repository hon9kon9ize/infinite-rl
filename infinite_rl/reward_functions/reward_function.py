from typing import Tuple


class RewardFunction:
    def __init__(self, task_name: str):
        self.task_name = task_name
        self.initialized = False

    def initialize(self):
        raise NotImplementedError("This method should be overridden by subclasses.")

    def compute_reward(
        self, model_output: str, reference_answer: str
    ) -> Tuple[float, float]:
        raise NotImplementedError("This method should be overridden by subclasses.")
