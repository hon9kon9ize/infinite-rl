import re
import json
from typing import Union, Callable
from sympy import simplify, parse_expr
from sympy.parsing.sympy_parser import (
    standard_transformations,
    implicit_multiplication_application,
)
from sympy.parsing.latex import parse_latex
from .reward_function import RewardFunction, RewardFunctionScore


class MathRewardFunction(RewardFunction):
    def __init__(self, task_name: str = "math", timeout: int = 5):
        super().__init__(task_name, timeout=timeout)

    def initialize(self):
        self.initialized = True

    def _check_equality(
        self, predicted_str: str, expected_str: str, allow_partial: bool = False
    ) -> float:
        """Simplified equality: extract first numeric value from strings and compare as floats.

        Behavior:
        - If both sides contain a parsable number, compare numerically (absolute tolerance 1e-9).
        - If parsing fails on either side, return 0.0 (no credit).
        - This intentionally ignores textual annotations/units: numbers only.
        """
        pred = predicted_str.strip()
        exp = expected_str.strip()

        # Try exact string match first (cheap)
        if pred.lower() == exp.lower():
            # If both strings are numeric-like, we'll check numerically below.
            pass

        # First: attempt symbolic equivalence using SymPy (handles expressions like (1/2)x^4 etc.)
        try:

            def to_sympy(text):
                text = re.sub(r"\+?\s*[cC]$", "", text).strip()
                if "\\" in text or "{" in text:
                    return parse_latex(text)
                else:
                    text = text.replace("^", "**")
                    transformations = standard_transformations + (
                        implicit_multiplication_application,
                    )
                    return parse_expr(text, transformations=transformations)

            pred_expr = to_sympy(pred)
            ref_expr = to_sympy(exp)
            if simplify(pred_expr - ref_expr) == 0:
                return 1.0
        except Exception:
            # If symbolic parsing fails, fall back to numeric-only comparison
            pass

        # Fallback: numeric-only comparison (extract numbers including fractions)
        def _extract_number(s: str):
            s_clean = s.replace("$", "").replace(",", "")
            s_clean = re.sub(r"\\text\{([^}]*)\}", r"\1", s_clean)
            s_clean = s_clean.replace("{", "").replace("}", "")

            # Fraction like 1/2
            m = re.search(r"([-+]?[0-9]*\.?[0-9]+)\s*/\s*([0-9]*\.?[0-9]+)", s_clean)
            if m:
                try:
                    num = float(m.group(1)) / float(m.group(2))
                    return num
                except Exception:
                    pass

            m2 = re.search(r"[-+]?[0-9]*\.?[0-9]+(?:[eE][-+]?[0-9]+)?", s_clean)
            if not m2:
                return None
            try:
                return float(m2.group(0))
            except Exception:
                return None

        p_num = _extract_number(pred)
        e_num = _extract_number(exp)

        if p_num is None or e_num is None:
            return 0.0

        return 1.0 if abs(p_num - e_num) <= 1e-9 else 0.0

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
            # Check for <answer> tags in expected output as well
            exp_matches = ExampleParser.extract_answer_tags(expected_output)
            if exp_matches:
                expected_output = exp_matches[0]

            trimmed = expected_output.strip()
            if trimmed.startswith("{") and trimmed.endswith("}"):
                try:
                    data = json.loads(trimmed)
                    for key in ["answer", "result", "solution", "value"]:
                        if key in data:
                            expected_output = data[key]
                            break
                except Exception:
                    pass

        matches = ExampleParser.extract_answer_tags(model_output, tags=answer_tag)
        if not matches:
            return RewardFunctionScore(
                format_score=0.0,
                correctness_score=0.0,
                error_msg="Missing <answer> tags in response.",
            )

        # Prepare expected list
        if isinstance(expected_output, (int, float)):
            expected_list = [str(expected_output)]
        elif callable(expected_output):
            # Special case: callable handles its own thing
            predicted_str = matches[0]
            try:
                result = expected_output(predicted_str)
                score = (
                    1.0
                    if (result is True)
                    else (result if isinstance(result, float) else 0.0)
                )
                return RewardFunctionScore(1.0, score)
            except Exception as e:
                return RewardFunctionScore(1.0, 0.0, f"Validator error: {e}")
        else:
            # String or other
            expected_str = str(expected_output).strip()
            # Handle joined answers from parser (now joined by newlines)
            if "\n" in expected_str:
                expected_list = [
                    p.strip() for p in expected_str.split("\n") if p.strip()
                ]
            elif " | " in expected_str:  # Backwards compatibility for older CSVs
                expected_list = [
                    p.strip() for p in expected_str.split("|") if p.strip()
                ]
            else:
                expected_list = [expected_str]

        # Validation logic
        # We now strictly enforce ONE <answer> tag containing a simple value.
        if len(matches) > 1:
            # Check if any of the tags match the expected answer
            # We still want to give some credit if the answer is there, but penalize format
            correctness = 0.0
            expected_str = str(expected_output).strip()
            for m in matches:
                if self._check_equality(m, expected_str) == 1.0:
                    correctness = 1.0
                    break

            return RewardFunctionScore(
                format_score=0.4,  # Heavy penalty for multiple tags
                correctness_score=correctness,
                error_msg="Multiple <answer> tags found. Math problems must have exactly one final answer tag.",
            )

        # Single match case
        predicted_str = matches[0]
        expected_str = str(expected_output).strip()
        allow_partial = isinstance(expected_output, (int, float))

        correctness = self._check_equality(
            predicted_str, expected_str, allow_partial=allow_partial
        )

        return RewardFunctionScore(
            format_score=1.0,
            correctness_score=correctness,
            error_msg=(
                ""
                if correctness == 1.0
                else f"Mathematical mismatch. Expected {expected_str}"
            ),
        )
