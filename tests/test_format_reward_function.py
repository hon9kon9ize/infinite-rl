import unittest
from infinite_rl.reward_functions.format import FormatRewardFunction


class TestFormatRewardFunction(unittest.TestCase):
    def test_python_good_block(self):
        fn = FormatRewardFunction(task_name="python")
        fn.initialize()
        out = "<answer>```python\nprint(1)```</answer>"
        score = fn.compute_reward(out, None)
        self.assertEqual(score.score, 1.0)

    def test_python_malformed_block(self):
        fn = FormatRewardFunction(task_name="python")
        fn.initialize()
        out = "<answer>```python\nprint(1)</answer>"  # missing closing backticks
        score = fn.compute_reward(out, None)
        self.assertEqual(score.score, 0.0)

    def test_math_simple_value(self):
        fn = FormatRewardFunction(task_name="math")
        fn.initialize()
        out = "<answer>12</answer>"
        score = fn.compute_reward(out, None)
        self.assertEqual(score.score, 1.0)

    def test_math_with_code_fence_invalid(self):
        fn = FormatRewardFunction(task_name="math")
        fn.initialize()
        out = "<answer>```python\n12```</answer>"
        score = fn.compute_reward(out, None)
        self.assertEqual(score.score, 0.0)


if __name__ == "__main__":
    unittest.main()
