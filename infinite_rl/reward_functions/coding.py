import json
import re
from typing import Union, Callable
from .reward_function import RewardFunction, RewardFunctionScore
from ..executor import Executor


class CodeRewardFunction(RewardFunction):
    """Base reward function for evaluating LLM-generated code solutions.

    This implementation is language-agnostic and expects a concrete subclass
    to set the `language` attribute via the constructor.
    """

    def __init__(
        self, task_name: str = "coding", language: str = "python", timeout: int = 5
    ):
        super().__init__(task_name, timeout=timeout)
        self.executor = None
        self.language = language.lower()  # e.g., 'python' or 'javascript'

    def initialize(self):
        """Initialize the executor for running code."""
        self.executor = Executor(timeout=self.timeout)
        self.initialized = True

    def compute_reward(
        self,
        model_output: str,
        expected_output: Union[str, int, Callable],
        answer_tag: str = "answer",
    ) -> RewardFunctionScore:
        from ..parser import ExampleParser

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

        # 1. Format Objective: Check for tags (default: <answer>)
        matches = ExampleParser.extract_answer_tags(model_output, tags=answer_tag)

        if not matches:
            return RewardFunctionScore(
                format_score=0.0,
                correctness_score=0.0,
                error_msg={
                    "coding": "Missing <answer> tags in response. Ensure the code is wrapped in <answer> and </answer>."
                },
            )

        content_to_parse = matches[0] if matches else ""
        format_score = 0.5  # Tag found

        # 1. Format Objective: Extract code from markdown block inside <answer>
        # We look for a code block matching the target language specifically first
        lang_pattern = rf"```(?:{self.language})\b\s*(.*?)```"
        match = re.search(lang_pattern, content_to_parse, re.DOTALL | re.IGNORECASE)

        if not match:
            # Fallback A: Generic code block with no language tag
            match = re.search(r"```(?!\w)\s*(.*?)```", content_to_parse, re.DOTALL)

        if not match:
            # Fallback B: Any tagged code block (e.g., if it used the wrong tag)
            match = re.search(r"```(?:\w+)?\s*(.*?)```", content_to_parse, re.DOTALL)

        if not match:
            # Fallback C: If triple backticks are present but we couldn't match a clean block,
            # or if they are malformed (e.g. tag inside block), try to clean it up.
            if "```" in content_to_parse:
                # If there's a trailing triple backtick that wasn't matched as a pair,
                # it's often because the model put the tag inside the block.
                # We'll try to strip it and treat the rest as code.
                cleaned_content = re.sub(
                    r"```(?:\w+)?\s*$", "", content_to_parse.strip()
                ).strip()
                cleaned_content = re.sub(
                    r"^```(?:\w+)?\s*", "", cleaned_content
                ).strip()

                if cleaned_content and len(cleaned_content) > 10:
                    code_content = cleaned_content
                    format_score = 0.3  # Penalize for malformed blocks
                else:
                    return RewardFunctionScore(
                        format_score=0.2,
                        correctness_score=0.0,
                        error_msg={
                            "coding": "Malformed markdown code block inside <answer> tags."
                        },
                    )
            else:
                # Fallback D: No triple backticks at all, treat the whole content as code
                code_content = content_to_parse
                format_score = 0.5  # Okay, but ideally should use code blocks
        else:
            code_content = match.group(1).strip()
            format_score = 1.0  # Tag + Code block successfully found

        # 2. Execution Objective: Try to execute the code
        try:
            stdout, stderr = self.executor.run_single(code_content, self.language)

            def filter_noise(text):
                if not text:
                    return ""
                noise_patterns = [
                    r"Debugger listening on.*",
                    r"For help, see:.*",
                    r"Debugger attached.*",
                    r"Waiting for the debugger to disconnect.*",
                    r"\(node:\d+\) [^:]+: .*",  # Node warnings
                    r"npm notice.*",  # npm noise
                ]
                filtered = text
                # Multi-line match for noise patterns
                for pattern in noise_patterns:
                    filtered = re.sub(
                        pattern, "", filtered, flags=re.IGNORECASE | re.MULTILINE
                    ).strip()
                return filtered

            # Filter out common noise from stderr (like Node.js debugger messages)
            filtered_stderr = filter_noise(stderr)
            if filtered_stderr:
                # Code extracted but failed during compilation or runtime
                # We penalize format_score to 0.5 as per expected behavior in tests
                return RewardFunctionScore(
                    format_score=0.5,
                    correctness_score=0.0,
                    error_msg={"coding": f"Execution failed:\n{stderr}"},
                )

            # Some runtimes may report failures to stdout instead of stderr (e.g., JS engines reporting ReferenceError)
            # Detect common error patterns in stdout and treat them as execution failures.
            error_patterns = [
                r"ReferenceError",
                r"SyntaxError",
                r"Error:",
                r"Traceback",
            ]
            if any(re.search(p, stdout, re.IGNORECASE) for p in error_patterns):
                return RewardFunctionScore(
                    format_score=0.5,
                    correctness_score=0.0,
                    error_msg={
                        "coding": f"Execution failed (reported in stdout):\n{stdout}"
                    },
                )

            # Execution was successful (no error patterns observed)
            # format_score remains what was set earlier (1.0 if perfectly tagged)

            # 3. Correctness Objective: Match stdout with expected_output

            # If expected_output is a string and contains code-like structure,
            # or it's wrapped in triple backticks, we should execute it to get the reference stdout.
            reference_stdout = ""

            # Robust extraction from Answer section: join ALL code blocks
            ref_code_blocks = re.findall(
                r"```(?:\w+)?\s*(.*?)```", str(expected_output), re.DOTALL
            )

            if ref_code_blocks:
                ref_code = "\n\n".join(ref_code_blocks).strip()
                ref_stdout, ref_stderr = self.executor.run_single(
                    ref_code, self.language
                )
                filtered_ref_stderr = filter_noise(ref_stderr)
                if filtered_ref_stderr:
                    return RewardFunctionScore(
                        format_score=format_score,
                        correctness_score=0.0,
                        error_msg={
                            "coding": f"Reference (Answer) execution failed: {ref_stderr}"
                        },
                    )
                reference_stdout = ref_stdout
            elif (
                any(
                    kw in str(expected_output)
                    for kw in ["def ", "class ", "import ", "print(", " = ", " ="]
                )
                or "(" in str(expected_output)
                and ")" in str(expected_output)  # Likely a function call
            ):
                # Heuristic: looks like raw code (maybe from <answer> tags without backticks)
                # First try to extract from <answer> if present
                from ..parser import ExampleParser

                ref_tags = ExampleParser.extract_answer_tags(str(expected_output))
                ref_code = ref_tags[0] if ref_tags else str(expected_output)

                # Cleanup potential trailing markdown artifacts
                ref_code = re.sub(r"```\s*$", "", ref_code).strip()

                ref_stdout, ref_stderr = self.executor.run_single(
                    ref_code, self.language
                )
                filtered_ref_stderr = filter_noise(ref_stderr)
                if filtered_ref_stderr:
                    # If it's a syntax error and we were just guessing it's code,
                    # maybe it's actually just text. But usually Answer is code.
                    # We only return 0 if there's a clear error and it looks like it SHOULD have been code.
                    if any(kw in ref_code for kw in ["def ", "class ", "import "]):
                        return RewardFunctionScore(
                            format_score=format_score,
                            correctness_score=0.0,
                            error_msg={
                                "coding": f"Reference (Answer) execution failed: {ref_stderr}"
                            },
                        )
                    # If it's ambiguous, fall back to comparing as string
                    reference_stdout = str(expected_output).strip()
                else:
                    reference_stdout = ref_stdout
            else:
                reference_stdout = str(expected_output).strip()

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
                        error_msg={"coding": f"Callable validator failed: {e}"},
                    )

            # Case B: String/Int/Other - require exact match
            # Compare trimmed stdout and reference_stdout for exact equality
            out_norm = stdout.strip()
            ref_norm = reference_stdout.strip()
            if out_norm == ref_norm:
                correctness_score = 1.0
                error_msg = {}
            else:
                correctness_score = 0.0
                error_msg = {
                    "coding": f"Output mismatch. Expected Output: {reference_stdout}\n"
                    f"Actual Output: {stdout}"
                }

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
                error_msg={"coding": f"Unexpected error: {str(e)}"},
            )


class PythonRewardFunction(CodeRewardFunction):
    """Reward function for Python code tasks."""

    def __init__(self, task_name: str = "python", timeout: int = 5):
        super().__init__(task_name=task_name, language="python", timeout=timeout)


class JavascriptRewardFunction(CodeRewardFunction):
    """Reward function for JavaScript code tasks."""

    def __init__(self, task_name: str = "javascript", timeout: int = 5):
        super().__init__(task_name=task_name, language="javascript", timeout=timeout)
