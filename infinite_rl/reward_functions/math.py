import re
from typing import TYPE_CHECKING
from .reward_function import RewardFunction, RewardFunctionScore

if TYPE_CHECKING:
    from ..task import Task


def _extract_number(s: str):
    """Extract numeric value from string, supporting fractions and decimals.

    Only accepts strings containing digits (0-9), slash (/), dot (.), and comma (,).
    Rejects any string with other characters including minus, plus, text, dollar signs, etc.

    Args:
        s: String potentially containing a number

    Returns:
        float if a number is found, None otherwise
    """
    s_clean = s.strip()

    # Only allow digits (0-9), slash (/), dot (.), comma (,), and whitespace
    # Reject any string containing other characters
    if not all(char.isdigit() or char in ["/", ".", ",", " "] for char in s_clean):
        return None

    # Remove commas for parsing (e.g., "1,000" -> "1000")
    s_clean = s_clean.replace(",", "")

    # Try to parse fraction like "1/2" or "3 / 4"
    m = re.search(r"([0-9]*\.?[0-9]+)\s*/\s*([0-9]*\.?[0-9]+)", s_clean)
    if m:
        try:
            return float(m.group(1)) / float(m.group(2))
        except Exception:
            pass

    # Try to parse regular number (decimals only, no scientific notation)
    m2 = re.search(r"[0-9]*\.?[0-9]+", s_clean)
    if m2:
        try:
            return float(m2.group(0))
        except Exception:
            pass

    return None


def _check_equality(predicted_str: str, expected_str: str) -> bool:
    """Compare two strings as numeric values.

    Args:
        predicted_str: Predicted answer string
        expected_str: Expected answer string

    Returns:
        True if both parse to numbers and are equal within tolerance, False otherwise
    """
    pred_num = _extract_number(predicted_str.strip())
    exp_num = _extract_number(expected_str.strip())

    if pred_num is None or exp_num is None:
        return False

    return abs(pred_num - exp_num) <= 1e-9


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

        expected_str = str(expected_output).strip()
        correctness = _check_equality(text, expected_str)

        return RewardFunctionScore(
            score=1.0 if correctness else 0.0,
            info=(
                ""
                if correctness
                else f"Mathematical mismatch. Expected {expected_str}, got {text}"
            ),
        )
