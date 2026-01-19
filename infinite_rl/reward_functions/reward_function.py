from typing import Union, Callable, Dict
from dataclasses import dataclass, field


@dataclass
class RewardFunctionScore:
    format_score: float
    correctness_score: float
    error_msg: Dict[str, str] = field(default_factory=dict)

    aux_score: float = 0.0


class RewardFunction:
    def __init__(self, task_name: str, timeout: int = 5):
        self.task_name = task_name
        self.timeout = timeout
        self.initialized = False

    def initialize(self):
        raise NotImplementedError("This method should be overridden by subclasses.")

    def compute_reward(
        self,
        model_output: str,
        expected_output: Union[str, int, float, Callable, None] = None,
        answer_tag: str = "answer",
    ) -> RewardFunctionScore:
        """Compute reward for given model output vs expected output.

        Parameters
        ----------
        model_output:
            Raw model response string.
        expected_output:
            Task-specific expected value (string, numeric, or validator). May be None.
        answer_tag:
            Optional tag name the model used to wrap the final output (default: 'answer').

        Notes
        -----
        Keeping the base method accepting **kwargs avoids forcing subclasses to
        declare every possible auxiliary parameter while allowing the orchestrator
        to pass additional context to functions that need it.
        """
        raise NotImplementedError("This method should be overridden by subclasses.")
