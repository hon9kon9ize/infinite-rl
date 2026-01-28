import json
import re
import subprocess
import os
import sys
from typing import Union, TYPE_CHECKING
from .reward_function import RewardFunction, RewardFunctionScore

if TYPE_CHECKING:
    from ..task import Task


class PuzzleRewardFunction(RewardFunction):
    """Reward function for evaluating LLM-generated sol functions against programming puzzles.

    The expected_output should be a dict with 'puzzle', 'inputs', and optionally 'language'.
    """

    def __init__(
        self,
        task_name: str = "puzzle",
        language: str = "python",
        timeout: int = 5,
        answer_tag: str = "answer",
        think_tag: str = "think",
    ):
        super().__init__(
            task_name, timeout=timeout, answer_tag=answer_tag, think_tag=think_tag
        )
        self.language = language.lower()  # e.g., 'python' or 'javascript'

    def initialize(self):
        """Initialize the reward function."""
        self.initialized = True

    def compute_reward(
        self,
        task: "Task",
        **kwargs,
    ) -> RewardFunctionScore:
        if not self.initialized:
            self.initialize()

        # Parse expected_output from task
        expected_output = task.expected_answer
        if isinstance(expected_output, str):
            try:
                expected_output = json.loads(expected_output)
            except:
                return RewardFunctionScore(
                    score=0.0, info="Invalid expected_output format"
                )

        puzzle_name = expected_output.get("puzzle")
        inputs = expected_output.get("inputs", {})
        language = expected_output.get("language", self.language)

        if not puzzle_name:
            return RewardFunctionScore(
                score=0.0,
                info="Missing puzzle name in expected_output",
            )

        # 1. Format Objective: Check for tags
        # First, check if raw answer tags exist
        tag_start = f"<{self.target_tag}>"
        tag_end = f"</{self.target_tag}>"
        if tag_start not in task.model_output or tag_end not in task.model_output:
            return RewardFunctionScore(
                score=0.0,
                info=f"Missing <{self.target_tag}> tags in response.",
            )

        # Extract raw content (with backticks still intact) to check format
        import re as re_module

        pattern = f"{re_module.escape(tag_start)}(.*?){re_module.escape(tag_end)}"
        raw_matches = re_module.findall(pattern, task.model_output, re_module.DOTALL)

        if not raw_matches:
            return RewardFunctionScore(
                score=0.0,
                info=f"Missing content in <{self.target_tag}> tags.",
            )

        raw_content = raw_matches[0]

        # Check for code block with language specifier in raw content
        lang_pattern = rf"```(?:{language})\b\s*(.*?)```"
        match = re_module.search(
            lang_pattern, raw_content, re_module.DOTALL | re_module.IGNORECASE
        )

        if not match:
            return RewardFunctionScore(
                score=0.0,
                info=f"Missing code block with language '{language}' inside <{self.target_tag}> tags.",
            )

        # 2. Extract sol function
        code_content = match.group(1).strip()

        # Check if sol function is defined
        sol_pattern = r"def sol\(|function sol\("
        if not re.search(sol_pattern, code_content):
            return RewardFunctionScore(
                score=0.0, info="Code must define a sol function"
            )

        # 3. Evaluate using runner.py for both languages
        runner_path = os.path.join(os.path.dirname(__file__), "..", "runner.py")
        puzzle_data = json.dumps(
            {
                "puzzle": puzzle_name,
                "inputs": inputs,
                "code": code_content,
                "language": language,
            }
        )
        try:
            result = subprocess.run(
                [sys.executable, runner_path],
                input=puzzle_data,
                text=True,
                capture_output=True,
                timeout=self.timeout,
            )
            if result.returncode != 0 or result.stderr:
                return RewardFunctionScore(
                    score=0.0,
                    info=f"Execution error: {result.stderr}",
                )
            output = json.loads(result.stdout.strip())
            if "error" in output:
                return RewardFunctionScore(
                    score=0.0,
                    info=f"Evaluation error: {output['error']}",
                )
            is_correct = output.get("isCorrect", False)
            if is_correct:
                return RewardFunctionScore(score=1.0)
            else:
                return RewardFunctionScore(
                    score=0.0,
                    info="Puzzle check failed",
                )
        except subprocess.TimeoutExpired:
            return RewardFunctionScore(score=0.0, info="Execution timed out")
        except Exception as e:
            return RewardFunctionScore(score=0.0, info=f"Evaluation failed: {str(e)}")
