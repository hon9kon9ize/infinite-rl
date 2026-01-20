import json
import re
from typing import Union, Callable
from .reward_function import RewardFunction, RewardFunctionScore
from ..executor import Executor
from ..utils.parser_utils import extract_tag


class CodeRewardFunction(RewardFunction):
    """Base reward function for evaluating LLM-generated code solutions.

    This implementation is language-agnostic and expects a concrete subclass
    to set the `language` attribute via the constructor.
    """

    def __init__(
        self,
        task_name: str = "coding",
        language: str = "python",
        timeout: int = 5,
        answer_tag: str = "answer",
        think_tag: str = "think",
    ):
        super().__init__(
            task_name, timeout=timeout, answer_tag=answer_tag, think_tag=think_tag
        )
        self.executor = None
        self.language = language.lower()  # e.g., 'python' or 'javascript'

    def initialize(self):
        """Initialize the executor for running code."""
        self.executor = Executor(timeout=self.timeout)
        self.initialized = True

    def compute_reward(
        self,
        model_output: str,
        expected_output: Union[str, int, float, None],
    ) -> RewardFunctionScore:
        if not self.initialized:
            self.initialize()

        # Handle expected_output being a JSON string (unwrap if necessary)
        if isinstance(expected_output, str):
            trimmed = expected_output.strip()
            if trimmed.startswith("{") and trimmed.endswith("}"):
                try:
                    data = json.loads(trimmed)
                    # Only unwrap if it's clearly an envelope for the code itself
                    for key in ["code", "solution"]:
                        if key in data:
                            expected_output = data[key]
                            break
                except Exception:
                    pass

        # 1. Format Objective: Check for tags using configured answer tag
        content_to_parse = extract_tag(model_output, tag=self.answer_tag)

        if not content_to_parse:
            return RewardFunctionScore(
                score=0.0,
                error_msg={
                    "coding": "Missing <answer> tags in response. Ensure the code is wrapped in <answer> and </answer>."
                },
            )

        # 1. Format Objective: Extract code from markdown block inside <answer>
        # We look for a code block matching the target language specifically first
        lang_pattern = rf"```(?:{self.language})\b\s*(.*?)```"
        match = re.search(lang_pattern, content_to_parse, re.DOTALL | re.IGNORECASE)

        if not match:
            return RewardFunctionScore(
                score=0.0,
                error_msg={
                    "coding": f"Missing code block with language '{self.language}' inside <answer> tags."
                },
            )

        # 2. Execution Objective: Try to execute the code
        code_content = match.group(1).strip()
        try:
            stdout, _stderr = self.executor.run_single(
                code_content, self.language
            )  # ignore stderr
        except Exception as e:
            # Catch-all for unexpected execution failures
            return RewardFunctionScore(
                score=0.0,
                error_msg={"coding": f"Unexpected error: {str(e)}"},
            )

        pred_answer = str(stdout).strip()
        exp_answer = str(expected_output).strip() if expected_output is not None else ""

        if pred_answer == exp_answer:
            return RewardFunctionScore(score=1.0)

        return RewardFunctionScore(
            score=0.0,
            error_msg={
                "coding": f"Output mismatch. Expected: '{exp_answer}', Got: '{pred_answer}'"
            },
        )


class PythonRewardFunction(CodeRewardFunction):
    """Reward function for Python code tasks."""

    def __init__(
        self,
        task_name: str = "python",
        timeout: int = 5,
        answer_tag: str = "answer",
        think_tag: str = "think",
    ):
        super().__init__(
            task_name=task_name,
            language="python",
            timeout=timeout,
            answer_tag=answer_tag,
            think_tag=think_tag,
        )


class JavascriptRewardFunction(CodeRewardFunction):
    """Reward function for JavaScript code tasks."""

    def __init__(
        self,
        task_name: str = "javascript",
        timeout: int = 5,
        answer_tag: str = "answer",
        think_tag: str = "think",
    ):
        super().__init__(
            task_name=task_name,
            language="javascript",
            timeout=timeout,
            answer_tag=answer_tag,
            think_tag=think_tag,
        )
