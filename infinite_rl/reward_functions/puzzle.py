import json
import re
import subprocess
import os
import sys
from typing import TYPE_CHECKING, Optional
from .reward_function import RewardFunction, RewardFunctionScore

if TYPE_CHECKING:
    from ..task import Task

# Module-level cache to avoid repeated subprocess overhead
_runner_process_cache: Optional[subprocess.Popen] = None


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

        # Special case for simulation dummy puzzle
        if puzzle_name == "dummy_puzzle":
            # For simulation, score based on format and code content
            import re as re_module

            # Find code block directly (prefer after </think>)
            lang_pattern = r"```(?:javascript)\b\s*(.*?)```"
            search_region = task.model_output
            think_close = f"</{self.think_tag}>"
            close_idx = task.model_output.find(think_close)
            if close_idx >= 0:
                search_region = task.model_output[close_idx + len(think_close):]

            match = re_module.search(
                lang_pattern, search_region, re_module.DOTALL | re_module.IGNORECASE
            )
            if not match:
                # Fallback: search entire output
                if close_idx >= 0:
                    match = re_module.search(
                        lang_pattern, task.model_output, re_module.DOTALL | re_module.IGNORECASE
                    )
            if not match:
                return RewardFunctionScore(
                    score=0.0,
                    info=f"Missing code block with language 'javascript' inside <{self.target_tag}> tags.",
                )
            code = match.group(1).strip()
            # For dummy, score 1.0 if "return true" in code, else 0.0
            if "return true" in code:
                return RewardFunctionScore(
                    score=1.0,
                    info="Dummy puzzle simulation - correct",
                )
            else:
                return RewardFunctionScore(
                    score=0.0,
                    info="Dummy puzzle simulation - incorrect",
                )

        # 1. Extract code block (```language ... ```)
        # Prefer code blocks AFTER </think> (reasoning boundary) to avoid prompt echoes.
        # Falls back to anywhere in output if no reasoning boundary found.
        import re as re_module

        lang_pattern = rf"```(?:{language})\b\s*(.*?)```"
        search_region = task.model_output
        think_close = f"</{self.think_tag}>"
        close_idx = task.model_output.find(think_close)
        if close_idx >= 0:
            search_region = task.model_output[close_idx + len(think_close):]

        match = re_module.search(
            lang_pattern, search_region, re_module.DOTALL | re_module.IGNORECASE
        )

        if not match:
            # Fallback: search entire output (in case no reasoning boundary)
            if close_idx >= 0:
                match = re_module.search(
                    lang_pattern, task.model_output, re_module.DOTALL | re_module.IGNORECASE
                )
                if not match:
                    return RewardFunctionScore(
                        score=0.0,
                        info=f"Missing code block with language '{language}' in response.",
                    )
            else:
                return RewardFunctionScore(
                    score=0.0,
                    info=f"Missing code block with language '{language}' in response.",
                )

        # 2. Extract sol function
        code_content = match.group(1).strip()

        # Check if sol function is defined
        sol_pattern = r"def sol\s*\(|function sol\s*\("
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
            # Use Popen with communicate() for better timeout handling
            # communicate() is more reliable than run() for timing out
            process = subprocess.Popen(
                [sys.executable, "-u", runner_path],  # -u for unbuffered output
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )

            try:
                stdout, stderr = process.communicate(
                    input=puzzle_data, timeout=self.timeout
                )
            except subprocess.TimeoutExpired:
                # Kill the process if it times out
                process.kill()
                process.wait(timeout=1)  # Wait for cleanup
                return RewardFunctionScore(
                    score=0.0, info=f"Execution timed out after {self.timeout}s"
                )

            # Only treat as error if return code is non-zero
            # Ignore stderr warnings (like wasmtime cleanup exceptions on Python 3.13)
            if process.returncode != 0:
                return RewardFunctionScore(
                    score=0.0,
                    info=f"Execution error (exit code {process.returncode}): {stderr}",
                )

            # Parse output even if there were warnings in stderr
            if not stdout or not stdout.strip():
                return RewardFunctionScore(
                    score=0.0,
                    info="No output from execution",
                )

            output = json.loads(stdout.strip())
            if "error" in output:
                error_msg = output["error"]
                if "stack" in output and output["stack"]:
                    error_msg += f"\nStack trace:\n{output['stack']}"
                return RewardFunctionScore(
                    score=0.0,
                    info=f"Evaluation error: {error_msg}",
                )
            is_correct = output.get("isCorrect", False)
            if is_correct:
                return RewardFunctionScore(score=1.0)
            else:
                return RewardFunctionScore(
                    score=0.0,
                    info="Puzzle check failed",
                )
        except Exception as e:
            return RewardFunctionScore(score=0.0, info=f"Evaluation failed: {str(e)}")
