import json
import re
import subprocess
import os
import sys
from typing import Union
from .reward_function import RewardFunction, RewardFunctionScore
from ..utils.parser_utils import extract_tag


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
        model_output: str,
        expected_output: Union[str, dict],
        target_tag: str = None,
    ) -> RewardFunctionScore:
        if not self.initialized:
            self.initialize()

        target_tag = target_tag if target_tag is not None else self.answer_tag

        # Parse expected_output
        if isinstance(expected_output, str):
            try:
                expected_output = json.loads(expected_output)
            except:
                return RewardFunctionScore(
                    score=0.0, error_msg={"puzzle": "Invalid expected_output format"}
                )

        puzzle_name = expected_output.get("puzzle")
        inputs = expected_output.get("inputs", {})
        language = expected_output.get("language", self.language)

        if not puzzle_name:
            return RewardFunctionScore(
                score=0.0,
                error_msg={"puzzle": "Missing puzzle name in expected_output"},
            )

        # 1. Format Objective: Check for tags
        content_to_parse = extract_tag(model_output, tag=target_tag)

        if not content_to_parse:
            return RewardFunctionScore(
                score=0.0,
                error_msg={"puzzle": f"Missing <{target_tag}> tags in response."},
            )

        # 1. Format Objective: Extract code from markdown block inside <answer>
        lang_pattern = rf"```(?:{language})\b\s*(.*?)```"
        match = re.search(lang_pattern, content_to_parse, re.DOTALL | re.IGNORECASE)

        if not match:
            return RewardFunctionScore(
                score=0.0,
                error_msg={
                    "puzzle": f"Missing code block with language '{language}' inside <{target_tag}> tags."
                },
            )

        # 2. Extract sol function
        code_content = match.group(1).strip()

        # Check if sol function is defined
        sol_pattern = r"def sol\(|function sol\("
        if not re.search(sol_pattern, code_content):
            return RewardFunctionScore(
                score=0.0, error_msg={"puzzle": "Code must define a sol function"}
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
                    error_msg={"puzzle": f"Execution error: {result.stderr}"},
                )
            output = json.loads(result.stdout.strip())
            if "error" in output:
                return RewardFunctionScore(
                    score=0.0,
                    error_msg={"puzzle": f"Evaluation error: {output['error']}"},
                )
            is_correct = output.get("isCorrect", False)
            if is_correct:
                return RewardFunctionScore(score=1.0)
            else:
                return RewardFunctionScore(
                    score=0.0,
                    error_msg={"puzzle": f"Puzzle check failed"},
                )
        except subprocess.TimeoutExpired:
            return RewardFunctionScore(
                score=0.0, error_msg={"puzzle": "Execution timed out"}
            )
        except Exception as e:
            return RewardFunctionScore(
                score=0.0, error_msg={"puzzle": f"Evaluation failed: {str(e)}"}
            )
