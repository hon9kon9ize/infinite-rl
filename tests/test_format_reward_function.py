import unittest
from infinite_rl.reward_functions.format import FormatRewardFunction
from infinite_rl.task import Task


class TestFormatRewardFunction(unittest.TestCase):
    def test_python_good_block(self):
        fn = FormatRewardFunction(task_name="python")
        fn.initialize()
        out = "<answer>```python\nprint(1)```</answer>"
        task = Task(
            task_id="test_1",
            task_name="test",
            task_type="puzzle",
            level=1,
            prompt="Test",
            expected_answer="",
            language="python",
            model_output=out,
        )
        score = fn.compute_reward(task)
        self.assertEqual(score.score, 1.0)

    def test_python_malformed_block(self):
        fn = FormatRewardFunction(task_name="python")
        fn.initialize()
        out = "<answer>```python\nprint(1)</answer>"  # missing closing backticks
        task = Task(
            task_id="test_2",
            task_name="test",
            task_type="puzzle",
            level=1,
            prompt="Test",
            expected_answer="",
            language="python",
            model_output=out,
        )
        score = fn.compute_reward(task)
        self.assertEqual(score.score, -1.0)

    def test_math_simple_value(self):
        fn = FormatRewardFunction(task_name="math")
        fn.initialize()
        out = "<answer>12</answer>"
        task = Task(
            task_id="test_3",
            task_name="test",
            task_type="math",
            level=1,
            prompt="Test",
            expected_answer="12",
            language="en",
            model_output=out,
        )
        score = fn.compute_reward(task)
        self.assertEqual(score.score, 1.0)

    def test_math_with_code_fence_invalid(self):
        fn = FormatRewardFunction(task_name="math")
        fn.initialize()
        out = "<answer>```python\n12```</answer>"
        task = Task(
            task_id="test_4",
            task_name="test",
            task_type="math",
            level=1,
            prompt="Test",
            expected_answer="12",
            language="en",
            model_output=out,
        )
        score = fn.compute_reward(task)
        self.assertEqual(score.score, -1.0)

    def test_content_before_opening_tag_think(self):
        """Test that content before <think> tag returns -1.0"""
        fn = FormatRewardFunction(task_name="format_think", target_tag="think")
        fn.initialize()
        out = """first think about the problem
here is my analysis

<think>
Now the actual thinking content
</think>

<answer>42</answer>"""
        task = Task(
            task_id="test_5",
            task_name="test",
            task_type="math",
            level=1,
            prompt="Test",
            expected_answer="42",
            language="en",
            model_output=out,
        )
        score = fn.compute_reward(task)
        self.assertEqual(score.score, -1.0)
        self.assertIn("before", score.info.lower())

    def test_content_before_opening_tag_answer(self):
        """Test that content before <answer> tag returns -1.0"""
        fn = FormatRewardFunction(task_name="format_answer", target_tag="answer")
        fn.initialize()
        out = """The answer is 42
<answer>42</answer>"""
        task = Task(
            task_id="test_6",
            task_name="test",
            task_type="math",
            level=1,
            prompt="Test",
            expected_answer="42",
            language="en",
            model_output=out,
        )
        score = fn.compute_reward(task)
        self.assertEqual(score.score, -1.0)
        self.assertIn("before", score.info.lower())

    def test_whitespace_before_tag_allowed(self):
        """Test that whitespace/newlines before tag is allowed"""
        fn = FormatRewardFunction(task_name="format_answer", target_tag="answer")
        fn.initialize()
        out = """

        
<answer>42</answer>"""
        task = Task(
            task_id="test_7",
            task_name="test",
            task_type="math",
            level=1,
            prompt="Test",
            expected_answer="42",
            language="en",
            model_output=out,
        )
        score = fn.compute_reward(task)
        self.assertEqual(score.score, 1.0)

    def test_nested_answer_inside_think_rejected(self):
        """Test that <answer> tag nested inside <think> tag is rejected"""
        fn = FormatRewardFunction(task_name="format_think", target_tag="think")
        fn.initialize()
        out = """<think>
Let me calculate this step by step.
The result is 108.
<answer>108</answer>
</think>"""
        task = Task(
            task_id="test_8",
            task_name="test",
            task_type="math",
            level=1,
            prompt="Test",
            expected_answer="108",
            language="en",
            model_output=out,
        )
        score = fn.compute_reward(task)
        self.assertEqual(score.score, -1.0)
        self.assertIn("nested", score.info.lower())
        self.assertIn("answer", score.info.lower())

    def test_nested_think_inside_answer_rejected(self):
        """Test that <think> tag nested inside <answer> tag is rejected"""
        fn = FormatRewardFunction(task_name="format_answer", target_tag="answer")
        fn.initialize()
        # Note: <answer> must be first for format_answer check to pass the "content before" check
        out = """<answer>
<think>Wait, let me reconsider...</think>
108
</answer>"""
        task = Task(
            task_id="test_9",
            task_name="test",
            task_type="math",
            level=1,
            prompt="Test",
            expected_answer="108",
            language="en",
            model_output=out,
        )
        score = fn.compute_reward(task)
        self.assertEqual(score.score, -1.0)
        self.assertIn("nested", score.info.lower())
        self.assertIn("think", score.info.lower())

    def test_nested_partial_tag_rejected(self):
        """Test that even opening tag without closing is rejected if nested"""
        fn = FormatRewardFunction(task_name="format_think", target_tag="think")
        fn.initialize()
        out = """<think>
My reasoning process
<answer>108
</think>"""
        task = Task(
            task_id="test_10",
            task_name="test",
            task_type="math",
            level=1,
            prompt="Test",
            expected_answer="108",
            language="en",
            model_output=out,
        )
        score = fn.compute_reward(task)
        self.assertEqual(score.score, -1.0)
        self.assertIn("nested", score.info.lower())

    def test_proper_structure_with_both_tags_accepted(self):
        """Test that proper structure with both tags in sequence is accepted"""
        # Test think tag validation - think must be first
        fn_think = FormatRewardFunction(task_name="format_think", target_tag="think")
        fn_think.initialize()
        out_think_first = """<think>
Let me solve this step by step.
First, I calculate 0.9 * 120 = 108
</think>

<answer>108</answer>"""
        task = Task(
            task_id="test_11a",
            task_name="test",
            task_type="math",
            level=1,
            prompt="Test",
            expected_answer="108",
            language="en",
            model_output=out_think_first,
        )
        score = fn_think.compute_reward(task)
        self.assertEqual(score.score, 1.0)

        # Test answer tag validation - when answer is first, it should also pass
        fn_answer = FormatRewardFunction(task_name="format_answer", target_tag="answer")
        fn_answer.initialize()
        out_answer_only = """<answer>108</answer>"""
        task2 = Task(
            task_id="test_11b",
            task_name="test",
            task_type="math",
            level=1,
            prompt="Test",
            expected_answer="108",
            language="en",
            model_output=out_answer_only,
        )
        score = fn_answer.compute_reward(task2)
        self.assertEqual(score.score, 1.0)

    def test_nested_closing_tag_only_rejected(self):
        """Test that closing tag of other type inside current tag is also rejected"""
        fn = FormatRewardFunction(task_name="format_think", target_tag="think")
        fn.initialize()
        out = """<think>
Some reasoning
</answer>
More reasoning
</think>

<answer>108</answer>"""
        task = Task(
            task_id="test_12",
            task_name="test",
            task_type="math",
            level=1,
            prompt="Test",
            expected_answer="108",
            language="en",
            model_output=out,
        )
        score = fn.compute_reward(task)
        self.assertEqual(score.score, -1.0)
        self.assertIn("nested", score.info.lower())

    def test_text_containing_tag_word_but_not_tag_accepted(self):
        """Test that text mentioning tag names (without brackets) is accepted"""
        fn = FormatRewardFunction(task_name="format_think", target_tag="think")
        fn.initialize()
        out = """<think>
I think the answer should be 108
Let me think about this more carefully
</think>

<answer>108</answer>"""
        task = Task(
            task_id="test_13",
            task_name="test",
            task_type="math",
            level=1,
            prompt="Test",
            expected_answer="108",
            language="en",
            model_output=out,
        )
        score = fn.compute_reward(task)
        self.assertEqual(score.score, 1.0)


if __name__ == "__main__":
    unittest.main()

