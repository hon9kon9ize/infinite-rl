import re
from typing import Union, Callable
from sympy import simplify, parse_expr
from sympy.parsing.sympy_parser import (
    standard_transformations,
    implicit_multiplication_application,
)
from sympy.parsing.latex import parse_latex
from .reward_function import RewardFunction


class MathRewardFunction(RewardFunction):
    def initialize(self):
        self.initialized = True

    def compute_reward(
        self, model_output: str, reference_answer: Union[str, int, Callable]
    ) -> float:
        tag_pattern = r"<answer>(.*?)</answer>"
        match = re.search(tag_pattern, model_output, re.DOTALL)
        if not match:
            return (0.0, 0.0)

        format_score = 1.0 if match else 0.0
        predicted_str = match.group(1).strip()

        # Handle different reference_answer types
        if callable(reference_answer):
            # Callable: pass prediction to validator function
            try:
                result = reference_answer(predicted_str)
                if isinstance(result, bool):
                    correctness_score = 1.0 if result else 0.0
                elif isinstance(result, float):
                    correctness_score = result
                else:
                    correctness_score = 0.0
                return (format_score, correctness_score)
            except Exception as e:
                print(f"Error executing validator: {e}")
                return (format_score, 0.0)

        elif isinstance(reference_answer, int):
            # Int: try to parse prediction as int and compare
            try:
                pred_int = int(float(predicted_str.strip()))
                if pred_int == reference_answer:
                    correctness_score = 1.0
                else:
                    # Partial credit based on closeness
                    diff = abs(pred_int - reference_answer)
                    correctness_score = max(
                        0.0, 1.0 - (diff / max(reference_answer, pred_int))
                    )
                return (format_score, correctness_score)
            except (ValueError, TypeError):
                return (format_score, 0.0)

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
                ref_expr = to_sympy(str(reference_answer))
                if simplify(pred_expr - ref_expr) == 0:
                    correctness_score = 1.0
                else:
                    correctness_score = 0.0
            except Exception:
                correctness_score = 0.0

            return (format_score, correctness_score)
