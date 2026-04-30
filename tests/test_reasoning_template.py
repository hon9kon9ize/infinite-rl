"""Unit tests for reasoning_template mode in reward functions.

When reasoning_template=True the model chat template auto-injects the opening think tag
so the model output omits it (closing tag is still present).

By setting reasoning_template=True reward functions extract reasoning content as
everything before the closing tag rather than requiring both opening and closing tags.
"""

import unittest
from infinite_rl.reward_functions.format import FormatRewardFunction
from infinite_rl.reward_functions.reasoning_steps import ReasoningStepsRewardFunction
from infinite_rl.reward_functions.length import LengthRewardFunction
from infinite_rl.reward_functions.reward_function import RewardFunction
from infinite_rl.task import Task

# Tags used in test model outputs
THINK_OPEN = "<think>"
THINK_CLOSE = "</think>"


class TestReasoningTemplateFormatReward(unittest.TestCase):
    """Test FormatRewardFunction with reasoning_template=True."""

    def test_valid_think_format(self):
        """Reasoning content before close tag is valid."""
        fn = FormatRewardFunction(
            task_name="format_think",
            target_tag="think",
            reasoning_template=True,
        )
        fn.initialize()
        task = Task(
            task_id="t1",
            task_name="test",
            task_type="math",
            level=0,
            prompt="Test",
            expected_answer="60",
            language="en",
            model_output="Let me solve this step by step." + THINK_CLOSE + "\n<answer>60</answer>",
        )
        score = fn.compute_reward(task)
        self.assertEqual(score.score, 1.0)
        self.assertIn("reasoning template", score.info)

    def test_valid_answer_format_with_reasoning_content(self):
        """Content before close tag is treated as valid reasoning for answer tag."""
        fn = FormatRewardFunction(
            task_name="format_answer",
            target_tag="answer",
            reasoning_template=True,
        )
        fn.initialize()
        task = Task(
            task_id="t2",
            task_name="test",
            task_type="math",
            level=0,
            prompt="Test",
            expected_answer="60",
            language="en",
            model_output="Reasoning here" + THINK_CLOSE + "\n<answer>60</answer>",
        )
        score = fn.compute_reward(task)
        self.assertEqual(score.score, 1.0)

    def test_missing_close_tag(self):
        """Missing close tag returns 0."""
        fn = FormatRewardFunction(
            task_name="format_think",
            target_tag="think",
            reasoning_template=True,
        )
        fn.initialize()
        task = Task(
            task_id="t3",
            task_name="test",
            task_type="math",
            level=0,
            prompt="Test",
            expected_answer="60",
            language="en",
            model_output="Just reasoning without closing tag\n<answer>60</answer>",
        )
        score = fn.compute_reward(task)
        self.assertEqual(score.score, 0.0)
        self.assertIn("closing tag", score.info.lower())

    def test_empty_content_before_close_tag(self):
        """Empty content before close tag returns 0."""
        fn = FormatRewardFunction(
            task_name="format_think",
            target_tag="think",
            reasoning_template=True,
        )
        fn.initialize()
        task = Task(
            task_id="t4",
            task_name="test",
            task_type="math",
            level=0,
            prompt="Test",
            expected_answer="60",
            language="en",
            model_output=THINK_CLOSE + "\n<answer>60</answer>",
        )
        score = fn.compute_reward(task)
        self.assertEqual(score.score, 0.0)
        self.assertIn("empty", score.info.lower())

    def test_multiple_close_tags(self):
        """Multiple close tags returns 0."""
        fn = FormatRewardFunction(
            task_name="format_think",
            target_tag="think",
            reasoning_template=True,
        )
        fn.initialize()
        task = Task(
            task_id="t5",
            task_name="test",
            task_type="math",
            level=0,
            prompt="Test",
            expected_answer="60",
            language="en",
            model_output="First" + THINK_CLOSE + "\nSecond" + THINK_CLOSE + "\n<answer>60</answer>",
        )
        score = fn.compute_reward(task)
        self.assertEqual(score.score, 0.0)
        self.assertIn("multiple", score.info.lower())

    def test_nested_answer_in_reasoning(self):
        """<answer> tag inside reasoning section is rejected."""
        fn = FormatRewardFunction(
            task_name="format_think",
            target_tag="think",
            reasoning_template=True,
        )
        fn.initialize()
        task = Task(
            task_id="t6",
            task_name="test",
            task_type="math",
            level=0,
            prompt="Test",
            expected_answer="60",
            language="en",
            model_output="The answer is <answer>60</answer>" + THINK_CLOSE + "\n<answer>60</answer>",
        )
        score = fn.compute_reward(task)
        self.assertEqual(score.score, 0.0)
        self.assertIn("nested", score.info.lower())

    def test_nested_answer_closing_in_reasoning(self):
        """</answer> tag inside reasoning section is rejected."""
        fn = FormatRewardFunction(
            task_name="format_think",
            target_tag="think",
            reasoning_template=True,
        )
        fn.initialize()
        task = Task(
            task_id="t7",
            task_name="test",
            task_type="math",
            level=0,
            prompt="Test",
            expected_answer="60",
            language="en",
            model_output="Some text </answer> more text" + THINK_CLOSE + "\n<answer>60</answer>",
        )
        score = fn.compute_reward(task)
        self.assertEqual(score.score, 0.0)
        self.assertIn("nested", score.info.lower())

    def test_standard_mode_rejected_without_open_tag(self):
        """Without reasoning_template missing open tag is rejected."""
        fn = FormatRewardFunction(
            task_name="format_think",
            target_tag="think",
            reasoning_template=False,
        )
        fn.initialize()
        task = Task(
            task_id="t8",
            task_name="test",
            task_type="math",
            level=0,
            prompt="Test",
            expected_answer="60",
            language="en",
            model_output="Reasoning without opening tag" + THINK_CLOSE + "\n<answer>60</answer>",
        )
        score = fn.compute_reward(task)
        self.assertEqual(score.score, 0.0)

    def test_standard_mode_accepted_with_both_tags(self):
        """Standard mode requires both open and close tags."""
        fn = FormatRewardFunction(
            task_name="format_think",
            target_tag="think",
            reasoning_template=False,
        )
        fn.initialize()
        task = Task(
            task_id="t9",
            task_name="test",
            task_type="math",
            level=0,
            prompt="Test",
            expected_answer="60",
            language="en",
            model_output=THINK_OPEN + "\nReasoning content\n" + THINK_CLOSE + "\n<answer>60</answer>",
        )
        score = fn.compute_reward(task)
        self.assertEqual(score.score, 1.0)


class TestReasoningTemplateReasoningSteps(unittest.TestCase):
    """Test ReasoningStepsRewardFunction with reasoning_template=True."""

    def test_indicators_found_in_reasoning_content(self):
        """Indicators in content before close tag are detected."""
        fn = ReasoningStepsRewardFunction(reasoning_template=True)
        fn.initialize()
        task = Task(
            task_id="t1",
            task_name="test",
            task_type="math",
            level=0,
            prompt="Test",
            expected_answer="60",
            language="en",
            model_output=(
                "First let me analyze the problem. "
                "Second I will compute the result. "
                "Finally the answer is clear."
                + THINK_CLOSE + "\n<answer>60</answer>"
            ),
        )
        score = fn.compute_reward(task)
        self.assertAlmostEqual(score.score, 1.0, places=5)

    def test_no_indicators_in_reasoning_content(self):
        """No indicators in content before close tag gets 0.0 (no penalty)."""
        fn = ReasoningStepsRewardFunction(reasoning_template=True)
        fn.initialize()
        task = Task(
            task_id="t2",
            task_name="test",
            task_type="math",
            level=0,
            prompt="Test",
            expected_answer="60",
            language="en",
            model_output=(
                "Just some random text without any markers"
                + THINK_CLOSE + "\n<answer>60</answer>"
            ),
        )
        score = fn.compute_reward(task)
        self.assertAlmostEqual(score.score, 0.0, places=5)

    def test_missing_close_tag_returns_zero(self):
        """Missing close tag returns 0."""
        fn = ReasoningStepsRewardFunction(reasoning_template=True)
        fn.initialize()
        task = Task(
            task_id="t3",
            task_name="test",
            task_type="math",
            level=0,
            prompt="Test",
            expected_answer="60",
            language="en",
            model_output="No closing tag at all\n<answer>60</answer>",
        )
        score = fn.compute_reward(task)
        self.assertAlmostEqual(score.score, 0.0, places=5)

    def test_standard_mode_rejected_without_open_tag(self):
        """Without reasoning_template missing open tag falls back to content before close tag."""
        fn = ReasoningStepsRewardFunction(reasoning_template=False)
        fn.initialize()
        task = Task(
            task_id="t4",
            task_name="test",
            task_type="math",
            level=0,
            prompt="Test",
            expected_answer="60",
            language="en",
            model_output=(
                "Reasoning without opening tag"
                + THINK_CLOSE + "\n<answer>60</answer>"
            ),
        )
        score = fn.compute_reward(task)
        # Falls back to content before close tag, finds 0 indicators
        self.assertAlmostEqual(score.score, 0.0, places=5)
        self.assertIn("indicators", score.info.lower())

    def test_standard_mode_works_with_both_tags(self):
        """Standard mode works with full open and close tags."""
        fn = ReasoningStepsRewardFunction(reasoning_template=False)
        fn.initialize()
        task = Task(
            task_id="t5",
            task_name="test",
            task_type="math",
            level=0,
            prompt="Test",
            expected_answer="60",
            language="en",
            model_output=(
                THINK_OPEN + "\nFirst step second step finally done.\n"
                + THINK_CLOSE + "\n<answer>60</answer>"
            ),
        )
        score = fn.compute_reward(task)
        self.assertAlmostEqual(score.score, 1.0, places=5)


class TestReasoningTemplateLength(unittest.TestCase):
    """Test LengthRewardFunction with reasoning_template=True."""

    def test_length_computed_from_reasoning_content(self):
        """Length is computed from content before close tag."""
        fn = LengthRewardFunction(task_name="length", reasoning_template=True)
        fn.initialize()
        task = Task(
            task_id="t1",
            task_name="test",
            task_type="math",
            level=0,
            prompt="Test",
            expected_answer="60",
            language="en",
            model_output="x" * 2000 + "\n" + THINK_CLOSE + "\n<answer>60</answer>",
        )
        score = fn.compute_reward(task)
        self.assertGreater(score.score, 0.5)
        self.assertLessEqual(score.score, 1.0)

    def test_too_short_reasoning_content(self):
        """Very short content before close tag gets low score."""
        fn = LengthRewardFunction(task_name="length", reasoning_template=True)
        fn.initialize()
        task = Task(
            task_id="t2",
            task_name="test",
            task_type="math",
            level=0,
            prompt="Test",
            expected_answer="60",
            language="en",
            model_output="short" + THINK_CLOSE + "\n<answer>60</answer>",
        )
        score = fn.compute_reward(task)
        self.assertAlmostEqual(score.score, 0.1, places=5)

    def test_missing_close_tag_returns_zero(self):
        """Missing close tag returns 0."""
        fn = LengthRewardFunction(task_name="length", reasoning_template=True)
        fn.initialize()
        task = Task(
            task_id="t3",
            task_name="test",
            task_type="math",
            level=0,
            prompt="Test",
            expected_answer="60",
            language="en",
            model_output="No closing tag\n<answer>60</answer>",
        )
        score = fn.compute_reward(task)
        self.assertAlmostEqual(score.score, 0.0, places=5)

    def test_standard_mode_works_with_both_tags(self):
        """Standard mode works with full open and close tags."""
        fn = LengthRewardFunction(task_name="length", reasoning_template=False)
        fn.initialize()
        task = Task(
            task_id="t4",
            task_name="test",
            task_type="math",
            level=0,
            prompt="Test",
            expected_answer="60",
            language="en",
            model_output=THINK_OPEN + "\n" + "x" * 2000 + "\n" + THINK_CLOSE + "\n<answer>60</answer>",
        )
        score = fn.compute_reward(task)
        self.assertGreater(score.score, 0.5)

    def test_standard_mode_rejected_without_open_tag(self):
        """Without reasoning_template missing open tag is rejected."""
        fn = LengthRewardFunction(task_name="length", reasoning_template=False)
        fn.initialize()
        task = Task(
            task_id="t5",
            task_name="test",
            task_type="math",
            level=0,
            prompt="Test",
            expected_answer="60",
            language="en",
            model_output=(
                "Reasoning without opening tag"
                + THINK_CLOSE + "\n<answer>60</answer>"
            ),
        )
        score = fn.compute_reward(task)
        self.assertAlmostEqual(score.score, 0.0, places=5)


class TestExtractThinkContent(unittest.TestCase):
    """Test RewardFunction.extract_think_content base method."""

    def test_reasoning_template_mode(self):
        """Extracts everything before close tag."""
        rf = RewardFunction(
            task_name="test",
            reasoning_template=True,
            think_tag="think",
        )
        output = "First step second step finally done.\n" + THINK_CLOSE + "\n<answer>42</answer>"
        content = rf.extract_think_content(output)
        self.assertIn("First step", content)
        self.assertIn("second step", content)
        self.assertNotIn(THINK_CLOSE, content)
        self.assertNotIn("<answer>", content)

    def test_standard_mode(self):
        """Extracts between open and close tags."""
        rf = RewardFunction(
            task_name="test",
            reasoning_template=False,
            think_tag="think",
        )
        output = THINK_OPEN + "\nReasoning content\n" + THINK_CLOSE + "\n<answer>42</answer>"
        content = rf.extract_think_content(output)
        self.assertEqual(content, "Reasoning content")

    def test_custom_tag(self):
        """Uses custom tag when provided."""
        rf = RewardFunction(
            task_name="test",
            reasoning_template=False,
            think_tag="think",
        )
        output = "<reasoning>Custom reasoning</reasoning>"
        content = rf.extract_think_content(output, tag="reasoning")
        self.assertEqual(content, "Custom reasoning")

    def test_reasoning_template_no_close_tag(self):
        """Returns empty when no closing tag found."""
        rf = RewardFunction(
            task_name="test",
            reasoning_template=True,
            think_tag="think",
        )
        output = "No closing tag here"
        content = rf.extract_think_content(output)
        self.assertEqual(content, "")

    def test_reasoning_template_custom_tag(self):
        """Uses custom tag for reasoning template mode."""
        rf = RewardFunction(
            task_name="test",
            reasoning_template=True,
            think_tag="think",
        )
        output = "Reasoning here</reasoning>\n<answer>42</answer>"
        content = rf.extract_think_content(output, tag="reasoning")
        self.assertEqual(content, "Reasoning here")


if __name__ == "__main__":
    unittest.main()
