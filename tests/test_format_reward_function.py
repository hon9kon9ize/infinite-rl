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
        self.assertEqual(score.score, 0.0)

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
        self.assertEqual(score.score, 0.0)


if __name__ == "__main__":
    unittest.main()
