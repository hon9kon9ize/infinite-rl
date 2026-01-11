import json
import re
from typing import Union, Callable
from .reward_function import RewardFunction, RewardFunctionScore
from ..executor import RewardExecutor


class CodingRewardFunction(RewardFunction):
    """Reward function for evaluating LLM-generated code solutions."""

    def __init__(self, task_name: str = "coding", timeout: int = 5):
        super().__init__(task_name, timeout=timeout)
        self.executor = None
        self.language = "python"  # default language

    def initialize(self):
        """Initialize the executor for running code."""
        self.executor = RewardExecutor(timeout=self.timeout)
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

        # 1. Format Objective: Check for <answer> tags
        matches = ExampleParser.extract_answer_tags(model_output)

        if not matches:
            return RewardFunctionScore(
                format_score=0.0,
                correctness_score=0.0,
                error_msg="Missing <answer> tags in response. Ensure the code is wrapped in <answer> and </answer>.",
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
                        error_msg="Malformed markdown code block inside <answer> tags.",
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
                    error_msg=f"Execution failed:\n{stderr}",
                )

            # Execution was successful (filtered stderr is empty)
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
                        error_msg=f"Reference (Answer) execution failed: {ref_stderr}",
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
                            error_msg=f"Reference (Answer) execution failed: {ref_stderr}",
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
        """
        if output1 is None:
            output1 = ""
        if output2 is None:
            output2 = ""

        o1 = output1.strip()
        o2 = output2.strip()
        print(f"DEBUG: SIM comparing '{o1}' with '{o2}'")

        if not o1 or not o2:
            return 1.0 if not o1 and not o2 else 0.0

        # 1. Exact match (case insensitive)
        if o1.lower() == o2.lower():
            return 1.0

        # 2. Advanced line-by-line normalization (handles different line endings, trailing spaces)
        def normalize_text(text):
            lines = text.replace("\r\n", "\n").split("\n")
            # Strip trailing space from each line and remove empty lines at end
            stripped_lines = [line.rstrip() for line in lines]
            return "\n".join(stripped_lines).strip()

        if normalize_text(o1).lower() == normalize_text(o2).lower():
            return 1.0

        # 3. Try JSON comparison
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

        # 4. Try numeric comparison
        try:
            f1 = float(o1.split()[-1]) if "=" in o1 else float(o1)
            f2 = float(o2.split()[-1]) if "=" in o2 else float(o2)
            if f1 == f2:
                return 1.0
            # Small tolerance for floats
            if abs(f1 - f2) < 1e-9:
                return 0.99
        except Exception:
            # Maybe the numeric part is embedded
            pass

        # 5. Normalized whitespace comparison
        if " ".join(o1.split()) == " ".join(o2.split()):
            return 0.95  # Increased from 0.9 since this is a very strong signal of equality

        # 5. Sequence similarity (respects order)
        # Use difflib for a robust similarity that penalizes wrong order
        import difflib

        return difflib.SequenceMatcher(None, o1.lower(), o2.lower()).ratio()
