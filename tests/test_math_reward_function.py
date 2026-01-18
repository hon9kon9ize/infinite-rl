import unittest
import json
from infinite_rl.reward_functions.math import MathRewardFunction


class TestMathRewardFunction(unittest.TestCase):
    """Test math reward function with SymPy evaluation."""

    def setUp(self):
        self.reward_fn = MathRewardFunction(task_name="math")
        self.reward_fn.initialize()

    def test_correct_integral_answer(self):
        """Test correct mathematical answer."""
        model_output = """
The integral of f(x) = 2x^3 - 4x + 1:

Using the power rule:
= 2x^4/4 - 4x^2/2 + x + C
= (1/2)x^4 - 2x^2 + x + C

<answer>(1/2)x^4 - 2x^2 + x + C</answer>
"""
        expected_output = "(1/2)x^4 - 2x^2 + x + C"

        score = self.reward_fn.compute_reward(model_output, expected_output)

        self.assertEqual(score.format_score, 1.0)  # Answer tag found
        self.assertEqual(score.correctness_score, 1.0)  # Mathematically correct

    def test_latex_answer_symbolic_match(self):
        """Latex formatted answer should match symbolic expected output."""
        model_output = """
The integral of f(x) = 2x^3 - 4x + 1 (LaTeX form):

Using the power rule:
= 2x^4/4 - 4x^2/2 + x + C
= \frac{1}{2}x^4 - 2x^2 + x + C

<answer>\\frac{1}{2}x^4 - 2x^2 + x + C</answer>
"""
        expected_output = "(1/2)x^4 - 2x^2 + x + C"

        score = self.reward_fn.compute_reward(model_output, expected_output)

        self.assertEqual(score.format_score, 1.0)
        self.assertEqual(score.correctness_score, 1.0)

    def test_missing_answer_tag(self):
        """Test when answer tag is missing."""
        model_output = "The answer is x^2 + 2x + 1 but no tags here."
        expected_output = "x^2 + 2x + 1"

        score = self.reward_fn.compute_reward(model_output, expected_output)

        self.assertEqual(score.format_score, 0.0)
        self.assertEqual(score.correctness_score, 0.0)

    def test_incorrect_mathematical_answer(self):
        """Test mathematically incorrect answer."""
        model_output = "<answer>x^4 - 2x^2 + x + C</answer>"
        expected_output = "(1/2)x^4 - 2x^2 + x + C"

        score = self.reward_fn.compute_reward(model_output, expected_output)

        self.assertEqual(score.format_score, 1.0)
        self.assertEqual(score.correctness_score, 0.0)

    def test_integer_expected_output(self):
        """Test with integer expected output."""
        model_output = "<answer>42</answer>"
        expected_output = 42

        score = self.reward_fn.compute_reward(model_output, expected_output)

        self.assertEqual(score.format_score, 1.0)
        self.assertEqual(score.correctness_score, 1.0)

    def test_numeric_only_comparison_from_string(self):
        """Ensure math reward compares numbers only when annotations are present."""
        cases = [
            ("<answer>10 \\text{hours}</answer>", 10),
            ("<answer>20\\,\\text{ml}</answer>", 20),
            ("<answer>50 \\text{ml/dose}</answer>", 50),
            ("<answer>$50</answer>", 50),
            ("<answer>$1000$</answer>", 1000),
            ("<answer>$1000</answer>", 1000),
            ("<answer>12 \\, \\text{cm|kg\\}</answer>", 12),
            ("<answer>3.14159</answer>", 3.14159),
        ]
        for model_out, expected in cases:
            score = self.reward_fn.compute_reward(model_out, expected)
            self.assertEqual(score.format_score, 1.0)
            self.assertEqual(score.correctness_score, 1.0)

    def test_non_numeric_fails(self):
        """If the answer can't be parsed to a number, correctness is 0.0."""
        score = self.reward_fn.compute_reward("<answer>ten hours</answer>", 10)
        self.assertEqual(score.format_score, 1.0)
        self.assertEqual(score.correctness_score, 0.0)

    def test_integer_with_wrong_value(self):
        """Test with integer but wrong value (now strict numeric equality)."""
        model_output = "<answer>40</answer>"
        expected_output = 42

        score = self.reward_fn.compute_reward(model_output, expected_output)

        self.assertEqual(score.format_score, 1.0)
        self.assertEqual(score.correctness_score, 0.0)

    def test_custom_tag_numeric_extraction(self):
        """Test numeric extraction when model uses a custom tag name."""
        model_output = "<result>100</result>"
        expected_output = 100

        score = self.reward_fn.compute_reward(
            model_output, expected_output, answer_tag="result"
        )

        self.assertEqual(score.format_score, 1.0)
        self.assertEqual(score.correctness_score, 1.0)
