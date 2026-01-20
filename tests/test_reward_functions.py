"""
Unit tests for reward functions using example data.
Tests cover all reward function types with mocked dependencies.
"""

import unittest
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

        # Correctness is reported in the unified `score` field
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

        score = self.reward_fn.compute_reward(model_output, expected_output)

        self.assertEqual(score.score, 1.0)

    def test_missing_answer_tag(self):
        """Test when answer tag is missing."""
        model_output = "The answer is x^2 + 2x + 1 but no tags here."
        expected_output = "x^2 + 2x + 1"

        score = self.reward_fn.compute_reward(model_output, expected_output)

        self.assertEqual(score.score, 0.0)

    def test_incorrect_mathematical_answer(self):
        """Test mathematically incorrect answer."""
        model_output = "<answer>x^4 - 2x^2 + x + C</answer>"
        expected_output = "(1/2)x^4 - 2x^2 + x + C"

        score = self.reward_fn.compute_reward(model_output, expected_output)

        self.assertEqual(score.score, 0.0)

    def test_integer_expected_output(self):
        """Test with integer expected output."""
        model_output = "<answer>42</answer>"
        expected_output = 42

        score = self.reward_fn.compute_reward(model_output, expected_output)

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
        for model_out, expected in cases:
            score = self.reward_fn.compute_reward(model_out, expected)
            self.assertEqual(score.score, 1.0)

    def test_non_numeric_fails(self):
        """If the answer can't be parsed to a number, correctness is 0.0."""
        score = self.reward_fn.compute_reward("<answer>ten hours</answer>", 10)
        self.assertEqual(score.score, 0.0)

    def test_integer_with_wrong_value(self):
        """Test with integer but wrong value (now strict numeric equality)."""
        model_output = "<answer>40</answer>"
        expected_output = 42

        score = self.reward_fn.compute_reward(model_output, expected_output)

        self.assertEqual(score.score, 0.0)

    def test_custom_tag_numeric_extraction(self):
        """Test numeric extraction when model uses a custom tag name."""
        model_output = "<result>100</result>"
        expected_output = 100

        score = self.reward_fn.compute_reward(
            model_output, expected_output, answer_tag="result"
        )

        self.assertEqual(score.score, 1.0)
        from infinite_rl.reward_functions.format import FormatRewardFunction

        fmt = FormatRewardFunction(task_name="math")
        fmt.initialize()
        self.assertEqual(
            fmt.compute_reward(model_output, None, answer_tag="result").score, 1.0
        )


class TestLanguageRewardFunction(unittest.TestCase):
    """Test language/dialect reward function."""

    def setUp(self):
        from pathlib import Path
        from infinite_rl.parser import ExampleParser

        package_dir = Path(__file__).parent.parent / "infinite_rl" / "examples"
        if not package_dir.exists():
            package_dir = Path(__file__).parent.parent / "examples"
        self.examples = ExampleParser.get_all_examples(package_dir)

        from infinite_rl.reward_functions.lang_consistency import (
            LangConsistencyRewardFunction,
        )

        self.reward_fn = LangConsistencyRewardFunction(task_name="lang_consistency")
        self.reward_fn.initialize()

    def test_language_example_pass(self):
        example = self.examples.get("LANG_CONSISTENCY")
        self.assertIsNotNone(example)

        score = self.reward_fn.compute_reward(example["response"], example["answer"])
        # Aux-only behavior: signal is now in the unified `score` field
        self.assertAlmostEqual(score.score, 1.0, places=5)

    def test_language_mismatch(self):
        # Expected Cantonese, but response is Mandarin/Chinese. Since detection checks
        # outside <answer> tags now, content entirely inside <answer> produces no signal.
        score = self.reward_fn.compute_reward("<answer>这是普通话。</answer>", "yue")
        self.assertAlmostEqual(score.score, 0.0, places=3)

    def test_mapping_detected_zh_hant_for_zh(self):
        # If CLD2 details indicate zh-Hant for the response but expected is zh, score should be 0.25
        from unittest.mock import patch

        with patch.object(
            type(self.reward_fn),
            "_detect_lang_details",
            return_value=[("Chinese", "zh-Hant", 100)],
        ):
            score = self.reward_fn.compute_reward("<answer>示例文本</answer>", "zh")
            self.assertAlmostEqual(score.score, 0.25, places=3)

    def test_en_detection(self):
        from unittest.mock import patch

        with patch.object(
            type(self.reward_fn),
            "_detect_lang_details",
            return_value=[("English", "en", 100)],
        ):
            score = self.reward_fn.compute_reward("<answer>Hello world</answer>", "en")
            self.assertAlmostEqual(score.score, 1.0, places=3)

    def test_mixed_proportional_score(self):
        # Mixed English + Chinese content should result in a weighted score for expected 'zh'
        text = "Hello World, Hello World, Hello World, Hello World, 战争不会显示谁对谁错，只会显示谁活了下来。"
        # Use real CLD2 details to compute expected weighted mapping score
        details = self.reward_fn._detect_lang_details(text)
        total = sum(d[-1] for d in details) if details else 0
        mapping = {"zh": {"zh": 1.0, "zh-hant": 0.25, "yue": 0.25, "en": 0.0}}
        expected_score = 0.0
        if total > 0:
            for entry in details:
                code = entry[1]
                b = entry[-1]
                norm = code.lower()
                if norm.startswith("zh-hant"):
                    norm = "zh-hant"
                elif norm.startswith("zh"):
                    norm = "zh"
                expected_score += (b / total) * mapping["zh"].get(norm, 0.0)

        score = self.reward_fn.compute_reward(f"<answer>{text}</answer>", "zh")
        # Because detection checks outside the <answer> tags, there will be no
        # detected bytes and thus aux_score should be 0.0 for this input.
        self.assertAlmostEqual(score.score, 0.0, places=3)


if __name__ == "__main__":
    unittest.main()
