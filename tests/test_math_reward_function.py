import unittest
import json
from infinite_rl.reward_functions.math import (
    MathRewardFunction,
    _last_boxed_only_string,
    _remove_boxed,
    _extract_boxed_answer,
    _extract_number,
    _to_sympy,
    _check_equality,
)
from infinite_rl.task import Task


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
        task = Task(
            task_id="test_1",
            task_name="test",
            task_type="math",
            level=1,
            prompt="Test",
            expected_answer=expected_output,
            language="en",
            model_output=model_output,
        )

        score = self.reward_fn.compute_reward(task)

        # Correctness reflected in unified `score` field
        self.assertEqual(score.score, 1.0)  # Mathematically correct

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
        task = Task(
            task_id="test_2",
            task_name="test",
            task_type="math",
            level=1,
            prompt="Test",
            expected_answer=expected_output,
            language="en",
            model_output=model_output,
        )

        score = self.reward_fn.compute_reward(task)

        self.assertEqual(score.score, 1.0)

    def test_missing_answer_tag(self):
        """Test when answer tag is missing."""
        model_output = "The answer is x^2 + 2x + 1 but no tags here."
        expected_output = "x^2 + 2x + 1"
        task = Task(
            task_id="test_3",
            task_name="test",
            task_type="math",
            level=1,
            prompt="Test",
            expected_answer=expected_output,
            language="en",
            model_output=model_output,
        )

        score = self.reward_fn.compute_reward(task)

        self.assertEqual(score.score, 0.0)

    def test_incorrect_mathematical_answer(self):
        """Test mathematically incorrect answer."""
        model_output = "<answer>x^4 - 2x^2 + x + C</answer>"
        expected_output = "(1/2)x^4 - 2x^2 + x + C"
        task = Task(
            task_id="test_4",
            task_name="test",
            task_type="math",
            level=1,
            prompt="Test",
            expected_answer=expected_output,
            language="en",
            model_output=model_output,
        )

        score = self.reward_fn.compute_reward(task)

        # Correctness now in unified `score` field; format checked separately
        self.assertEqual(score.score, 0.0)

    def test_integer_expected_output(self):
        """Test with integer expected output."""
        model_output = "<answer>42</answer>"
        expected_output = 42
        task = Task(
            task_id="test_5",
            task_name="test",
            task_type="math",
            level=1,
            prompt="Test",
            expected_answer=expected_output,
            language="en",
            model_output=model_output,
        )

        score = self.reward_fn.compute_reward(task)

        self.assertEqual(score.score, 1.0)

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
        for i, (model_out, expected) in enumerate(cases):
            task = Task(
                task_id=f"test_{i}",
                task_name="test",
                task_type="math",
                level=1,
                prompt="Test",
                expected_answer=expected,
                language="en",
                model_output=model_out,
            )
            score = self.reward_fn.compute_reward(task)
            self.assertEqual(score.score, 1.0)

    def test_non_numeric_fails(self):
        """If the answer can't be parsed to a number, correctness is 0.0."""
        task = Task(
            task_id="test_6",
            task_name="test",
            task_type="math",
            level=1,
            prompt="Test",
            expected_answer=10,
            language="en",
            model_output="<answer>ten hours</answer>",
        )
        score = self.reward_fn.compute_reward(task)
        self.assertEqual(score.score, 0.0)

    def test_integer_with_wrong_value(self):
        """Test with integer but wrong value (now strict numeric equality)."""
        model_output = "<answer>40</answer>"
        expected_output = 42
        task = Task(
            task_id="test_7",
            task_name="test",
            task_type="math",
            level=1,
            prompt="Test",
            expected_answer=expected_output,
            language="en",
            model_output=model_output,
        )

        score = self.reward_fn.compute_reward(task)

        self.assertEqual(score.score, 0.0)


class TestMathHelperFunctions(unittest.TestCase):
    """Test helper functions in math reward module."""

    def test_last_boxed_only_string_with_boxed(self):
        """Test extracting last boxed string."""
        text = "Some text \\boxed{answer1} and \\boxed{answer2}"
        result = _last_boxed_only_string(text)
        self.assertEqual(result, "\\boxed{answer2}")

    def test_last_boxed_only_string_with_fbox(self):
        """Test extracting last fbox string."""
        text = "Some text \\fbox{answer1} and \\fbox{answer2}"
        result = _last_boxed_only_string(text)
        self.assertEqual(result, "\\fbox{answer2}")

    def test_last_boxed_only_string_no_boxed(self):
        """Test when no boxed content exists."""
        text = "Some text without boxed content"
        result = _last_boxed_only_string(text)
        self.assertIsNone(result)

    def test_last_boxed_only_string_unclosed_boxed(self):
        """Test when boxed content is not properly closed."""
        text = "Some text \\boxed{unclosed"
        result = _last_boxed_only_string(text)
        self.assertIsNone(result)

    def test_remove_boxed_valid(self):
        """Test removing boxed wrapper from valid input."""
        text = "\\boxed{answer}"
        result = _remove_boxed(text)
        self.assertEqual(result, "answer")

    def test_remove_boxed_invalid(self):
        """Test removing boxed wrapper from invalid input."""
        text = "not boxed"
        result = _remove_boxed(text)
        self.assertIsNone(result)

    def test_extract_boxed_answer(self):
        """Test extracting boxed answer from text."""
        text = "Some text \\boxed{42} and more"
        result = _extract_boxed_answer(text)
        self.assertEqual(result, "42")

    def test_extract_number_integer(self):
        """Test extracting integer from string."""
        result = _extract_number("42")
        self.assertEqual(result, 42.0)

    def test_extract_number_float(self):
        """Test extracting float from string."""
        result = _extract_number("3.14159")
        self.assertEqual(result, 3.14159)

    def test_extract_number_fraction(self):
        """Test extracting fraction from string."""
        result = _extract_number("1/2")
        self.assertEqual(result, 0.5)

    def test_extract_number_invalid_fraction(self):
        """Test extracting invalid fraction falls back to None."""
        result = _extract_number("a/b")
        self.assertIsNone(result)

    def test_extract_number_with_text(self):
        """Test extracting number with surrounding text."""
        result = _extract_number("The answer is 42 hours")
        self.assertEqual(result, 42.0)

    def test_extract_number_with_dollar(self):
        """Test extracting number with dollar signs."""
        result = _extract_number("$1000")
        self.assertEqual(result, 1000.0)

    def test_extract_number_with_commas(self):
        """Test extracting number with commas."""
        result = _extract_number("1,000,000")
        self.assertEqual(result, 1000000.0)

    def test_extract_number_scientific(self):
        """Test extracting scientific notation."""
        result = _extract_number("1.23e-4")
        self.assertEqual(result, 1.23e-4)

    def test_extract_number_invalid(self):
        """Test extracting from invalid string."""
        result = _extract_number("not a number")
        self.assertIsNone(result)

    def test_to_sympy_latex(self):
        """Test converting LaTeX to sympy."""
        result = _to_sympy("\\frac{1}{2}")
        self.assertEqual(str(result), "1/2")

    def test_to_sympy_regular(self):
        """Test converting regular expression to sympy."""
        result = _to_sympy("x^2 + 2*x + 1")
        self.assertEqual(str(result), "x**2 + 2*x + 1")

    def test_check_equality_numeric(self):
        """Test numeric equality checking."""
        result = _check_equality("42", "42")
        self.assertTrue(result)

    def test_check_equality_symbolic(self):
        """Test symbolic equality checking."""
        result = _check_equality("x^2 + 2*x + 1", "(x + 1)^2")
        self.assertTrue(result)

    def test_check_equality_unequal(self):
        """Test inequality detection."""
        result = _check_equality("42", "43")
        self.assertFalse(result)

    def test_math_reward_with_boxed_expected(self):
        """Test math reward when expected output contains boxed answer."""
        reward_fn = MathRewardFunction()
        reward_fn.initialize()

        model_output = "<answer>42</answer>"
        expected_output = "\\boxed{42}"
        task = Task(
            task_id="test_8",
            task_name="test",
            task_type="math",
            level=1,
            prompt="Test",
            expected_answer=expected_output,
            language="en",
            model_output=model_output,
        )

        score = reward_fn.compute_reward(task)
        self.assertEqual(score.score, 1.0)

    def test_math_reward_with_boxed_model_and_expected(self):
        """Test math reward when both model and expected contain boxed answers."""
        reward_fn = MathRewardFunction()
        reward_fn.initialize()

        model_output = "<answer>\\boxed{42}</answer>"
        expected_output = "\\boxed{42}"
        task = Task(
            task_id="test_9",
            task_name="test",
            task_type="math",
            level=1,
            prompt="Test",
            expected_answer=expected_output,
            language="en",
            model_output=model_output,
        )

        score = reward_fn.compute_reward(task)
        self.assertEqual(score.score, 1.0)
