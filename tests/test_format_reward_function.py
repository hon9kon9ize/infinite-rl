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


if __name__ == "__main__":
    unittest.main()
