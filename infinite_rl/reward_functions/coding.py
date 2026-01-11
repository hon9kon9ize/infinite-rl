from typing import Union, Callable
from .reward_function import RewardFunction, RewardFunctionScore
from ..executor import RewardExecutor
import json
import re


class CodingRewardFunction(RewardFunction):
    """Reward function for evaluating LLM-generated code solutions."""

    def __init__(self, task_name: str = "coding"):
        super().__init__(task_name)
        self.executor = None
        self.language = "python"  # default language

    def initialize(self):
        """Initialize the executor for running code."""
        self.executor = RewardExecutor(timeout=5)
        self.initialized = True

    def set_language(self, language: str):
        """Set the programming language for code execution."""
        supported_langs = [
            "python",
            "javascript",
            "js",
            "typescript",
            "ts",
            "cpp",
            "c++",
            "rust",
            "java",
        ]
        if language.lower() not in supported_langs:
            raise ValueError(f"Unsupported language: {language}")
        self.language = language.lower()

    def compute_reward(
        self,
        model_output: str,
        expected_output: Union[str, int, Callable],
    ) -> RewardFunctionScore:
        """
        Compute reward for generated code with robust extraction and matching.
        """
        if not self.initialized:
            self.initialize()

        # 1. Format Objective: Extract code from markdown block
        # We look for a code block matching the target language specifically first
        lang_pattern = rf"```(?:{self.language})\b\s*(.*?)```"
        match = re.search(lang_pattern, model_output, re.DOTALL | re.IGNORECASE)

        if not match:
            # Fallback A: Generic code block with no language tag
            match = re.search(r"```(?!\w)\s*(.*?)```", model_output, re.DOTALL)

        if not match:
            # Fallback B: Any tagged code block (e.g., if it used the wrong tag)
            match = re.search(r"```(?:\w+)?\s*(.*?)```", model_output, re.DOTALL)

        if not match:
            return RewardFunctionScore(
                format_score=0.0,
                correctness_score=0.0,
                error_msg="No markdown code block found in response.",
            )

        code_content = match.group(1).strip()
        format_score = 0.5  # Code block successfully found

        # 2. Execution Objective: Try to execute the code
        try:
            stdout, stderr = self.executor.run_single(code_content, self.language)

            if stderr:
                # Code extracted but failed during compilation or runtime
                return RewardFunctionScore(
                    format_score=format_score,
                    correctness_score=0.0,
                    error_msg=f"Execution failed:\n{stderr}",
                )

            # Execution was successful (no stderr)
            format_score = 1.0

            # 3. Correctness Objective: Match stdout with expected_output

            # If expected_output is a string and contains code-like structure,
            # or it's wrapped in triple backticks, we should execute it to get the reference stdout.
            reference_stdout = str(expected_output)

            # Check if expected_output is actually a code block
            code_match = re.search(
                r"```(?:\w+)?\s*(.*?)```", str(expected_output), re.DOTALL
            )
            if code_match:
                ref_code = code_match.group(1).strip()
                ref_stdout, ref_stderr = self.executor.run_single(
                    ref_code, self.language
                )
                if ref_stderr:
                    return RewardFunctionScore(
                        format_score=format_score,
                        correctness_score=0.0,
                        error_msg=f"Reference (Answer) execution failed: {ref_stderr}",
                    )
                reference_stdout = ref_stdout
            elif (
                "def " in str(expected_output)
                or "class " in str(expected_output)
                or "import " in str(expected_output)
            ):
                # Heuristic: looks like raw code
                ref_stdout, ref_stderr = self.executor.run_single(
                    str(expected_output), self.language
                )
                if ref_stderr:
                    return RewardFunctionScore(
                        format_score=format_score,
                        correctness_score=0.0,
                        error_msg=f"Reference (Answer) execution failed: {ref_stderr}",
                    )
                reference_stdout = ref_stdout

            # Case A: Callable validator
            if callable(expected_output):
                try:
                    result = expected_output(stdout)
                    if isinstance(result, bool):
                        correctness_score = 1.0 if result else 0.0
                    elif isinstance(result, (int, float)):
                        correctness_score = float(result)
                    else:
                        correctness_score = 0.0
                    return RewardFunctionScore(
                        format_score=format_score, correctness_score=correctness_score
                    )
                except Exception as e:
                    return RewardFunctionScore(
                        format_score=format_score,
                        correctness_score=0.0,
                        error_msg=f"Callable validator failed: {e}",
                    )

            # Case B: String/Int/Other - use robust similarity matching
            similarity = self._compute_similarity(stdout, reference_stdout)
            correctness_score = similarity

            error_msg = ""
            if (
                correctness_score < 0.5
            ):  # Changed from 1.0 to 0.5 for less noise in error messages
                error_msg = (
                    f"Output mismatch. Similarity: {similarity:.2f}.\n"
                    f"Expected Output: {reference_stdout}\n"
                    f"Actual Output: {stdout}"
                )

            return RewardFunctionScore(
                format_score=format_score,
                correctness_score=correctness_score,
                error_msg=error_msg,
            )

        except Exception as e:
            # Catch-all for unexpected execution failures
            return RewardFunctionScore(
                format_score=format_score,
                correctness_score=0.0,
                error_msg=f"Unexpected error: {str(e)}",
            )

    def _compute_similarity(self, output1: str, output2: str) -> float:
        """
        Robust comparison between two output strings.
        Strategies:
        1. Exact match (case-insensitive, trimmed)
        2. JSON structure comparison (if both are valid JSON)
        3. Numeric comparison (if both are numbers)
        4. Token overlap (Jaccard similarity)
        """
        o1 = output1.strip()
        o2 = output2.strip()

        if not o1 or not o2:
            return 1.0 if not o1 and not o2 else 0.0

        # 1. Exact match (case insensitive)
        if o1.lower() == o2.lower():
            return 1.0

        # 2. Try JSON comparison
        try:
            # Try to see if both are valid JSON objects/lists
            if (o1.startswith("{") or o1.startswith("[")) and (
                o2.startswith("{") or o2.startswith("[")
            ):
                j1 = json.loads(o1)
                j2 = json.loads(o2)
                if j1 == j2:
                    return 1.0
        except Exception:
            pass

        # 3. Try numeric comparison
        try:
            f1 = float(o1)
            f2 = float(o2)
            if f1 == f2:
                return 1.0
            # Small tolerance for floats
            if abs(f1 - f2) < 1e-9:
                return 0.99
        except Exception:
            pass

        # 4. Normalized whitespace comparison
        if " ".join(o1.split()) == " ".join(o2.split()):
            return 0.9

        # 5. Sequence similarity (respects order)
        # Use difflib for a robust similarity that penalizes wrong order
        import difflib

        return difflib.SequenceMatcher(None, o1.lower(), o2.lower()).ratio()
