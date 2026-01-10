import re
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

    def compute_reward(self, model_output: str, reference_answer: str) -> float:
        tag_pattern = r"<answer>(.*?)</answer>"
        match = re.search(tag_pattern, model_output, re.DOTALL)
        if not match:
            return (0.0, 0.0)
        format_score = 1.0 if match else 0.0
        predicted_str = match.group(1).strip()
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
            ref_expr = to_sympy(reference_answer)
            if simplify(pred_expr - ref_expr) == 0:
                correctness_score = 1.0
            else:
                correctness_score = 0.0
        except Exception:
            correctness_score = 0.0
        return (format_score, correctness_score)
