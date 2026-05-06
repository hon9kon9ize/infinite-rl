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

    def to_dict(self) -> dict:
        """Convert to dict for logging."""
        return {
            "score": self.score,
            "reward_function_name": self.reward_function_name,
            "info": self.info,
        }


class RewardFunction:
    def __init__(
        self,
        task_name: str,
        timeout: int = 5,
        answer_tag: str = "answer",
        think_tag: str = "think",
        target_tag: str = None,
        reasoning_template: bool = False,
    ):
        self.task_name = task_name
        self.timeout = timeout
        self.initialized = False
        # Default tags for parsing structured answers and reasoning hints
        self.answer_tag = answer_tag
        self.think_tag = think_tag
        self.target_tag = target_tag if target_tag is not None else answer_tag
        # When True, model's chat template already injects <think>.
        # Output omits the opening <think> tag (closing </think> is still present).
        self.reasoning_template = reasoning_template

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

    def extract_think_content(self, model_output: str, tag: str = None) -> str:
        """Extract reasoning (think) content from model output.

        When `reasoning_template` is True, the opening <think> tag is omitted
        by the chat template, so we extract everything before </think> instead.

        Parameters
        ----------
        model_output:
            Raw model response string.
        tag:
            Tag to use for extraction. Defaults to `self.think_tag`.

        Returns
        -------
        Extracted think content as string (empty if not found).
        """
        target = tag if tag is not None else self.think_tag
        if self.reasoning_template:
            # Chat template injected </think> — content is everything before </think>
            think_close = f"</{target}>"
            close_index = model_output.find(think_close)
            if close_index > 0:
                content = model_output[:close_index].strip()
                # Some collapsed checkpoints still emit a literal opening tag
                # even when the chat template already injected it. Do not let
                # the tag itself count as reasoning content.
                think_open = f"<{target}>"
                if content.startswith(think_open):
                    content = content[len(think_open):].strip()
                return content
            return ""
        else:
            # Standard: extract between tags
            content = extract_tag(
                model_output,
                tag=target,
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
