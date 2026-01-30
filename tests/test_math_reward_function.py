import unittest
from infinite_rl.reward_functions.math import (
    MathRewardFunction,
    _extract_number,
    _check_equality,
)
from infinite_rl.task import Task


class TestMathRewardFunction(unittest.TestCase):
    """Test math reward function with numeric comparison only."""

    def setUp(self):
        self.reward_fn = MathRewardFunction(task_name="math")
        self.reward_fn.initialize()

    def test_correct_numeric_answer(self):
        """Test correct numeric answer."""
        model_output = "<answer>42</answer>"
        expected_output = "42"
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
        self.assertEqual(score.score, 1.0)

    def test_correct_fractional_answer(self):
        """Test correct fractional answer (1/2)."""
        model_output = "<answer>0.5</answer>"
        expected_output = "1/2"
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
        model_output = "<answer>40</answer>"
        expected_output = "42"
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
            ("<answer>10</answer>", 10),
            ("<answer>20</answer>", 20),
            ("<answer>50</answer>", 50),
            ("<answer>$50</answer>", 50),
            ("<answer>$1000$</answer>", 1000),
            ("<answer>$1000</answer>", 1000),
            ("<answer>12</answer>", 12),
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

    def test_check_equality_numeric(self):
        """Test numeric equality checking."""
        result = _check_equality("42", "42")
        self.assertTrue(result)

    def test_check_equality_fraction_vs_decimal(self):
        """Test fraction vs decimal equality checking."""
        result = _check_equality("1/2", "0.5")
        self.assertTrue(result)

    def test_check_equality_unequal(self):
        """Test inequality detection."""
        result = _check_equality("42", "43")
        self.assertFalse(result)

    def test_check_equality_invalid_input(self):
        """Test equality checking with invalid input."""
        result = _check_equality("abc", "42")
        self.assertFalse(result)

    def test_math_reward_simple_number(self):
        """Test math reward with simple numeric answer."""
        reward_fn = MathRewardFunction()
        reward_fn.initialize()

        model_output = "<answer>42</answer>"
        expected_output = "42"
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

    def test_math_reward_with_text_in_answer(self):
        """Test math reward when answer contains text (extracts first number)."""
        reward_fn = MathRewardFunction()
        reward_fn.initialize()

        model_output = "<answer>The answer is 42 units</answer>"
        expected_output = "42"
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
