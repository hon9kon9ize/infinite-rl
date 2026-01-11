import re
from typing import Union, Callable
from sympy import simplify, parse_expr
from sympy.parsing.sympy_parser import (
    standard_transformations,
    implicit_multiplication_application,
)
from sympy.parsing.latex import parse_latex
from .reward_function import RewardFunction, RewardFunctionScore


class MathRewardFunction(RewardFunction):
    def initialize(self):
        self.initialized = True

    def compute_reward(
        self, model_output: str, expected_output: Union[str, int, Callable]
    ) -> RewardFunctionScore:
        tag_pattern = r"<answer>(.*?)</answer>"
        match = re.search(tag_pattern, model_output, re.DOTALL)
        if not match:
            return RewardFunctionScore(format_score=0.0, correctness_score=0.0)

        format_score = 1.0 if match else 0.0
        predicted_str = match.group(1).strip()

        # Handle different expected_output types
        if callable(expected_output):
            # Callable: pass prediction to validator function
            try:
                result = expected_output(predicted_str)
                if isinstance(result, bool):
                    correctness_score = 1.0 if result else 0.0
                elif isinstance(result, float):
                    correctness_score = 1.0 if result > 0.5 else 0.0
                else:
                    correctness_score = 0.0
                return RewardFunctionScore(
                    format_score=format_score, correctness_score=correctness_score
                )
            except Exception as e:
                print(f"Error executing validator: {e}")
                return RewardFunctionScore(
                    format_score=format_score, correctness_score=0.0
                )

        elif isinstance(expected_output, int):
            # Int: try to parse prediction as int and compare
            try:
                pred_int = int(float(predicted_str.strip()))
                if pred_int == expected_output:
                    correctness_score = 1.0
                else:
                    # Partial credit based on closeness
                    diff = abs(pred_int - expected_output)
                    similarity = max(0.0, 1.0 - (diff / max(expected_output, pred_int)))
                    correctness_score = 1.0 if similarity > 0.5 else 0.0
                return RewardFunctionScore(
                    format_score=format_score, correctness_score=correctness_score
                )
            except (ValueError, TypeError):
                return RewardFunctionScore(
                    format_score=format_score, correctness_score=0.0
                )

        else:
            # String: use existing symbolic math comparison
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

                pred_expr = to_sympy(predicted_str)
                ref_expr = to_sympy(str(expected_output))
                if simplify(pred_expr - ref_expr) == 0:
                    correctness_score = 1.0
                else:
                    correctness_score = 0.0
            except Exception:
                correctness_score = 0.0

            return RewardFunctionScore(
                format_score=format_score, correctness_score=correctness_score
            )
