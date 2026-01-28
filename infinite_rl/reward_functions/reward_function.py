from typing import Union, Callable, Dict, TYPE_CHECKING
from dataclasses import dataclass, field

from ..utils.parser_utils import extract_tag

if TYPE_CHECKING:
    from ..task import Task


@dataclass
class RewardFunctionScore:
    score: float
    reward_function_name: str = ""
    info: str = ""


class RewardFunction:
    def __init__(
        self,
        task_name: str,
        timeout: int = 5,
        answer_tag: str = "answer",
        think_tag: str = "think",
        target_tag: str = None,
    ):
        self.task_name = task_name
        self.timeout = timeout
        self.initialized = False
        # Default tags for parsing structured answers and reasoning hints
        self.answer_tag = answer_tag
        self.think_tag = think_tag
        self.target_tag = target_tag if target_tag is not None else answer_tag

    def initialize(self):
        raise NotImplementedError("This method should be overridden by subclasses.")

    def extract_tag(
        self,
        model_output: str,
    ) -> str:
        """Extract target content from model output using specified tag.

        Parameters
        ----------
        model_output:
            Raw model response string.
        target_tag:
            Tag to extract from model output. If None, uses `self.answer_tag`.
        Returns
        -------
        Extracted content as string.
        """
        content = extract_tag(
            model_output,
            tag=self.target_tag,
        ).strip()
        return content

    def compute_reward(
        self,
        task: "Task",
        **kwargs,
    ) -> RewardFunctionScore:
        """Compute reward for given model output vs expected output.

        Parameters
        ----------
        task:
            Task object containing expected_answer, language, and other metadata

        Notes
        -----
        Tag handling (for example `answer_tag` and `think_tag`) is managed at the
        instance level by the RewardFunction (set via the constructor). Do not
        pass tag parameters to this method; use instance attributes instead.
        """
        raise NotImplementedError("This method should be overridden by subclasses.")
