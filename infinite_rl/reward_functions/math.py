import re
from typing import Union, TYPE_CHECKING
from .reward_function import RewardFunction, RewardFunctionScore

if TYPE_CHECKING:
    from ..task import Task


def _last_boxed_only_string(string):
    idx = string.rfind("\\boxed")
    if idx < 0:
        idx = string.rfind("\\fbox")
        if idx < 0:
            return None

    i = idx
    right_brace_idx = None
    num_left_braces_open = 0
    while i < len(string):
        if string[i] == "{":
            num_left_braces_open += 1
        if string[i] == "}":
            num_left_braces_open -= 1
            if num_left_braces_open == 0:
                right_brace_idx = i
                break
        i += 1

    if right_brace_idx is None:
        retval = None
    else:
        retval = string[idx : right_brace_idx + 1]

    return retval


def _remove_boxed(s):
    left = "\\boxed{"
    try:
        assert s[: len(left)] == left
        assert s[-1] == "}"
        return s[len(left) : -1]
    except AssertionError:
        return None


def _extract_boxed_answer(s: str):
    """Extract the answer from inside a LaTeX \\boxed{} command"""
    solution = _last_boxed_only_string(s)
    solution = _remove_boxed(solution)
    return solution


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


def _to_sympy(text):
    from sympy import parse_expr
    from sympy.parsing.sympy_parser import (
        standard_transformations,
        implicit_multiplication_application,
    )
    from sympy.parsing.latex import parse_latex

    text = re.sub(r"\+?\s*[cC]$", "", text).strip()
    if "\\" in text or "{" in text:
        return parse_latex(text)
    else:
        text = text.replace("^", "**")
        transformations = standard_transformations + (
            implicit_multiplication_application,
        )
        return parse_expr(text, transformations=transformations)


def _check_equality(predicted_str: str, expected_str: str) -> float:
    """Simplified equality: extract first numeric value from strings and compare as floats.

    Behavior:
    - If both sides contain a parsable number, compare numerically (absolute tolerance 1e-9).
    - If parsing fails on either side, return 0.0 (no credit).
    - This intentionally ignores textual annotations/units: numbers only.
    """
    from sympy import simplify

    pred = predicted_str.strip()
    exp = expected_str.strip()

    # Try exact string match first (cheap)
    if pred.lower() == exp.lower():
        # If both strings are numeric-like, we'll check numerically below.
        pass

    # First: attempt symbolic equivalence using SymPy (handles expressions like (1/2)x^4 etc.)
    try:
        pred_expr = _to_sympy(pred)
        ref_expr = _to_sympy(exp)
        if simplify(pred_expr - ref_expr) == 0:
            return True
    except Exception:
        # If symbolic parsing fails, fall back to numeric-only comparison
        pass

    # Fallback: numeric-only comparison (extract numbers including fractions)
    p_num = _extract_number(pred)
    e_num = _extract_number(exp)

    if p_num is None or e_num is None:
        return False

    return True if abs(p_num - e_num) <= 1e-9 else False


class MathRewardFunction(RewardFunction):
    def __init__(
        self,
        task_name: str = "math",
        timeout: int = 5,
        answer_tag: str = "answer",
        think_tag: str = "think",
    ):
        super().__init__(
            task_name, timeout=timeout, answer_tag=answer_tag, think_tag=think_tag
        )

    def initialize(self):
        self.initialized = True

    def compute_reward(
        self,
        task: "Task",
        **kwargs,
    ) -> RewardFunctionScore:
        if not self.initialized:
            self.initialize()

        expected_output = task.expected_answer
        text = self.extract_tag(task.model_output or "")

        if not text:
            return RewardFunctionScore(
                score=0.0,
                info=f"No answer found in the <{self.target_tag}> tag.",
            )

        if "\\boxed" in text:
            text = _extract_boxed_answer(text)

        expected_str = str(expected_output).strip()

        if "\\boxed" in expected_str:
            expected_str = _extract_boxed_answer(expected_str)

        correctness = _check_equality(text, expected_str)

        return RewardFunctionScore(
            score=1.0 if correctness else 0.0,
            info=(
                ""
                if correctness == 1.0
                else f"Mathematical mismatch. Expected {expected_str}"
            ),
        )
