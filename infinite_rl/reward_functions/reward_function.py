from typing import Union, Callable, Dict
from dataclasses import dataclass, field


@dataclass
class RewardFunctionScore:
    score: float
    error_msg: Dict[str, str] = field(default_factory=dict)


class RewardFunction:
    def __init__(
        self,
        task_name: str,
        timeout: int = 5,
        answer_tag: str = "answer",
        think_tag: str = "think",
    ):
        self.task_name = task_name
        self.timeout = timeout
        self.initialized = False
        # Default tags for parsing structured answers and reasoning hints
        self.answer_tag = answer_tag
        self.think_tag = think_tag

    def initialize(self):
        raise NotImplementedError("This method should be overridden by subclasses.")

    def compute_reward(
        self,
        model_output: str,
        expected_output: Union[str, int, float, None] = None,
    ) -> RewardFunctionScore:
        """Compute reward for given model output vs expected output.

        Parameters
        ----------
        model_output:
            Raw model response string.
        expected_output:
            Task-specific expected value (string, numeric, or validator). May be None.

        Notes
        -----
        Tag handling (for example `answer_tag` and `think_tag`) is managed at the
        instance level by the RewardFunction (set via the constructor). Do not
        pass tag parameters to this method; use instance attributes instead.
        """
        raise NotImplementedError("This method should be overridden by subclasses.")
