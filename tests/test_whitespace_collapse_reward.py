"""Unit tests for WhitespaceCollapseRewardFunction."""

import unittest
from infinite_rl.reward_functions.whitespace_collapse import (
    WhitespaceCollapseRewardFunction,
)
from infinite_rl.task import Task


class TestWhitespaceCollapseRewardFunction(unittest.TestCase):
    """Test whitespace collapse detection reward function."""

    def setUp(self):
        """Set up test fixtures."""
        self.reward_fn = WhitespaceCollapseRewardFunction(reasoning_language="en")
        self.reward_fn.initialize()

    def test_normal_spacing_math_task(self):
        """Test that normally spaced math reasoning gets no penalty."""
        task = Task(
            task_id="math_test_1",
            task_name="Normal spacing math",
            task_type="math",
            level=0,
            prompt="What is 2+2?",
            expected_answer="4",
            language="en",
            reasoning_language="en",
            model_output=(
                "<think>Let me think about this carefully. I need to add two numbers together. "
                "Two plus two means I start with two and add two more to it. The result is four.</think>"
                "<answer>4</answer>"
            ),
        )
        result = self.reward_fn.compute_reward(task)
        self.assertEqual(result.score, 0.0, "Normal spacing should not be penalized")
        self.assertIn("Normal spacing", result.info)

    def test_collapsed_spacing_math_task(self):
        """Test that space-collapsed reasoning is penalized."""
        # Create text with no spaces (0% space ratio)
        collapsed_text = "Thequickbrownfoxjumpsoverthelazydog." * 4
        task = Task(
            task_id="math_test_2",
            task_name="Collapsed spacing math",
            task_type="math",
            level=0,
            prompt="What is 2+2?",
            expected_answer="4",
            language="en",
            reasoning_language="en",
            model_output=f"<think>{collapsed_text}</think><answer>4</answer>",
        )
        result = self.reward_fn.compute_reward(task)
        self.assertEqual(
            result.score, -0.5, "Collapsed spacing should be penalized with -0.5"
        )
        self.assertIn("Whitespace collapse detected", result.info)

    def test_puzzle_with_english_reasoning(self):
        """Test puzzle task with English reasoning in think tag."""
        task = Task(
            task_id="puzzle_js_1",
            task_name="JavaScript puzzle with English reasoning",
            task_type="puzzle",
            level=1,
            prompt="Sort the array",
            expected_answer="",
            language="javascript",  # programming language
            reasoning_language="en",  # reasoning language is English
            model_output=(
                "<think>I need to implement a sorting algorithm. "
                "The quicksort algorithm is efficient for this task.</think>"
                "<answer>function sort(arr) { return arr.sort(); }</answer>"
            ),
        )
        result = self.reward_fn.compute_reward(task)
        self.assertEqual(
            result.score,
            0.0,
            "Normal English reasoning in puzzle should not be penalized",
        )

    def test_puzzle_with_collapsed_english_reasoning(self):
        """Test puzzle task with collapsed English reasoning."""
        collapsed_text = "Thequickbrownfoxjumpsoverthelazydog." * 4
        task = Task(
            task_id="puzzle_js_2",
            task_name="JavaScript puzzle with collapsed reasoning",
            task_type="puzzle",
            level=1,
            prompt="Sort the array",
            expected_answer="",
            language="javascript",  # programming language
            reasoning_language="en",  # reasoning language is English
            model_output=f"<think>{collapsed_text}</think><answer>function sort(arr) {{ return arr.sort(); }}</answer>",
        )
        result = self.reward_fn.compute_reward(task)
        self.assertEqual(
            result.score, -0.5, "Collapsed English reasoning should be penalized"
        )
        self.assertIn("Whitespace collapse detected", result.info)

    def test_non_english_reasoning_skipped(self):
        """Test that non-English reasoning is skipped."""
        task = Task(
            task_id="math_test_3",
            task_name="Chinese math",
            task_type="math",
            level=0,
            prompt="What is 2+2?",
            expected_answer="4",
            language="zh",  # Chinese
            reasoning_language="zh",  # Chinese reasoning
            model_output=(
                "<think>这是一个中文句子。没有空格的文本。这是另一个中文句子。"
                "没有空格的文本。这是另一个中文句子。没有空格的文本。</think>"
                "<answer>4</answer>"
            ),
        )
        result = self.reward_fn.compute_reward(task)
        self.assertEqual(result.score, 0.0, "Non-English reasoning should be skipped")
        self.assertIn("only checking English", result.info)

    def test_short_reasoning_skipped(self):
        """Test that short reasoning text is skipped."""
        task = Task(
            task_id="math_test_4",
            task_name="Short reasoning",
            task_type="math",
            level=0,
            prompt="What is 2+2?",
            expected_answer="4",
            language="en",
            reasoning_language="en",
            model_output="<think>Quick answer</think><answer>4</answer>",
        )
        result = self.reward_fn.compute_reward(task)
        self.assertEqual(result.score, 0.0, "Short reasoning should be skipped")
        self.assertIn("Text too short", result.info)

    def test_no_think_tag(self):
        """Test that output without think tag is skipped."""
        task = Task(
            task_id="math_test_5",
            task_name="No think tag",
            task_type="math",
            level=0,
            prompt="What is 2+2?",
            expected_answer="4",
            language="en",
            reasoning_language="en",
            model_output="<answer>4</answer>",
        )
        result = self.reward_fn.compute_reward(task)
        self.assertEqual(
            result.score, 0.0, "Output without think tag should be skipped"
        )
        self.assertIn("No <think> tag found", result.info)

    def test_no_model_output(self):
        """Test that missing model output is handled gracefully."""
        task = Task(
            task_id="math_test_6",
            task_name="No output",
            task_type="math",
            level=0,
            prompt="What is 2+2?",
            expected_answer="4",
            language="en",
            reasoning_language="en",
            model_output=None,
        )
        result = self.reward_fn.compute_reward(task)
        self.assertEqual(result.score, 0.0, "Missing output should be skipped")
        self.assertIn("No output to analyze", result.info)

    def test_borderline_spacing_ratio(self):
        """Test text at the boundary of space ratio threshold."""
        # Create text with exactly 5% spaces (just at threshold)
        # For 1000 chars, 5% = 50 spaces
        text_with_spaces = (
            " word" * 50
        ) + "a" * 750  # 50 spaces, 950 non-space = 950 chars
        # Verify our calculation
        space_count = text_with_spaces.count(" ")
        text_len = len(text_with_spaces)
        space_ratio = space_count / text_len
        # Should be around 5%
        self.assertGreaterEqual(space_ratio, 0.04)
        self.assertLessEqual(space_ratio, 0.06)

        task = Task(
            task_id="math_test_7",
            task_name="Borderline spacing",
            task_type="math",
            level=0,
            prompt="Test",
            expected_answer="test",
            language="en",
            reasoning_language="en",
            model_output=f"<think>{text_with_spaces}</think><answer>test</answer>",
        )
        result = self.reward_fn.compute_reward(task)
        # At 5% space ratio (at the threshold), should be penalized since < 0.05 triggers it
        self.assertEqual(
            result.score, -0.5, "At threshold spacing ratio should trigger penalty"
        )

    def test_custom_space_ratio_threshold(self):
        """Test custom space ratio threshold."""
        # Create text with 3% spaces (below default 5% threshold)
        # 30 spaces in 1000 chars = 3%
        text_with_spaces = (" word" * 30) + "a" * 850  # 30 spaces, 1000 chars = 3%
        space_ratio = text_with_spaces.count(" ") / len(text_with_spaces)
        self.assertLess(space_ratio, 0.05)

        # Test 1: Default threshold (5%) should penalize 3% spacing
        reward_fn_default = WhitespaceCollapseRewardFunction(reasoning_language="en")
        reward_fn_default.initialize()
        task = Task(
            task_id="math_test_8a",
            task_name="Custom threshold default",
            task_type="math",
            level=0,
            prompt="Test",
            expected_answer="test",
            language="en",
            reasoning_language="en",
            model_output=f"<think>{text_with_spaces}</think><answer>test</answer>",
        )
        result_default = reward_fn_default.compute_reward(task)
        self.assertEqual(
            result_default.score, -0.5, "Default threshold should penalize <5% spacing"
        )

        # Test 2: Custom threshold (2%) should not penalize 3% spacing
        reward_fn_custom = WhitespaceCollapseRewardFunction(
            reasoning_language="en", space_ratio_threshold=0.02
        )
        reward_fn_custom.initialize()
        result_custom = reward_fn_custom.compute_reward(task)
        self.assertEqual(
            result_custom.score,
            0.0,
            "Custom 2% threshold should not penalize 3% spacing",
        )

    def test_reasoning_language_defaults_to_language(self):
        """Test that reasoning_language defaults to language if not specified."""
        task = Task(
            task_id="math_test_9",
            task_name="Default reasoning language",
            task_type="math",
            level=0,
            prompt="What is 2+2?",
            expected_answer="4",
            language="en",
            # reasoning_language not specified, should default to language
            model_output=(
                "<think>Two plus two makes four. This is simple arithmetic.</think>"
                "<answer>4</answer>"
            ),
        )
        # Verify the default was applied
        self.assertEqual(task.reasoning_language, "en")
        result = self.reward_fn.compute_reward(task)
        self.assertEqual(result.score, 0.0, "Default reasoning language should work")

    def test_puzzle_language_field_ignored(self):
        """Test that puzzle.language (programming language) doesn't affect reasoning check."""
        # Puzzle with Python programming language but English reasoning
        task = Task(
            task_id="puzzle_py_1",
            task_name="Python puzzle",
            task_type="puzzle",
            level=2,
            prompt="Implement factorial",
            expected_answer="",
            language="python",  # This is the programming language
            reasoning_language="en",  # This is the reasoning language
            model_output=(
                "<think>Factorial is the product of all positive integers up to n. "
                "I need to implement this recursively or iteratively.</think>"
                "<answer>def factorial(n):\n    return n * factorial(n-1) if n > 1 else 1</answer>"
            ),
        )
        result = self.reward_fn.compute_reward(task)
        self.assertEqual(
            result.score,
            0.0,
            "Python programming language should not affect English reasoning check",
        )


if __name__ == "__main__":
    unittest.main()
