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
        """Helper to check equality between two strings using exact, numeric, and symbolic methods."""
        pred = predicted_str.strip()
        exp = expected_str.strip()

        # 1. Exact match
        if pred.lower() == exp.lower():
            return 1.0

        # 2. Numeric match
        try:
            p_clean = pred.replace("$", "").replace(",", "")
            e_clean = exp.replace("$", "").replace(",", "")
            p_val = float(p_clean)
            e_val = float(e_clean)
            if abs(p_val - e_val) < 1e-6:
                return 1.0

            if allow_partial:
                # Partial credit based on closeness
                diff = abs(p_val - e_val)
                denominator = max(abs(e_val), abs(p_val))
                if denominator > 0:
                    return max(0.0, 1.0 - (diff / denominator))
        except (ValueError, TypeError):
            pass

        # 3. Symbolic match
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
            pass

        return 0.0

    def compute_reward(
        self, model_output: str, expected_output: Union[str, int, Callable]
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

        matches = ExampleParser.extract_answer_tags(model_output)
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
