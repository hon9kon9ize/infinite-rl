import unittest
from infinite_rl.reward_functions.reasoning_steps import ReasoningStepsRewardFunction
from infinite_rl.task import Task


class TestReasoningStepsRewardFunction(unittest.TestCase):
    def setUp(self):
        self.rf = ReasoningStepsRewardFunction()
        self.rf.initialize()

    def test_missing_think_tag(self):
        task = Task(
            task_id="test_1",
            task_name="test",
            task_type="puzzle",
            level=1,
            prompt="Test",
            expected_answer="answer",
            language="en",
            model_output="I think the answer is 2.",
        )
        out = self.rf.compute_reward(task)
        self.assertEqual(out.score, 0.0)

    def test_single_indicator(self):
        model_output = "<think>First, we compute the sum.</think>"
        task = Task(
            task_id="test_2",
            task_name="test",
            task_type="puzzle",
            level=1,
            prompt="Test",
            expected_answer="answer",
            language="en",
            model_output=model_output,
        )
        out = self.rf.compute_reward(task)
        # Single indicator gives 0.5 bonus
        self.assertAlmostEqual(out.score, 0.5, places=5)

    def test_multiple_unique_indicators(self):
        model_output = "<think>First, we compute. Second, we verify. Finally, we present the result.</think>"
        task = Task(
            task_id="test_3",
            task_name="test",
            task_type="puzzle",
            level=1,
            prompt="Test",
            expected_answer="answer",
            language="en",
            model_output=model_output,
        )
        out = self.rf.compute_reward(task)
        # Three or more indicators (>=3) give 1.0 bonus
        self.assertAlmostEqual(out.score, 1.0, places=5)

    def test_two_indicators(self):
        model_output = "<think>First, we compute. Second, we verify.</think>"
        task = Task(
            task_id="test_3b",
            task_name="test",
            task_type="puzzle",
            level=1,
            prompt="Test",
            expected_answer="answer",
            language="en",
            model_output=model_output,
        )
        out = self.rf.compute_reward(task)
        # Exactly two indicators give 0.7 bonus
        self.assertAlmostEqual(out.score, 0.7, places=5)
        model_output = "<think>First. First. First.</think>"
        task = Task(
            task_id="test_4",
            task_name="test",
            task_type="puzzle",
            level=1,
            prompt="Test",
            expected_answer="answer",
            language="en",
            model_output=model_output,
        )
        out = self.rf.compute_reward(task)
        # Single unique indicator gives 0.5 bonus
        self.assertAlmostEqual(out.score, 0.5, places=5)

    def test_no_indicators_penalty(self):
        """Test that no reasoning indicators results in 0.0 (no penalty)."""
        model_output = "<think>The answer is correct.</think>"
        task = Task(
            task_id="test_5",
            task_name="test",
            task_type="puzzle",
            level=1,
            prompt="Test",
            expected_answer="answer",
            language="en",
            model_output=model_output,
        )
        out = self.rf.compute_reward(task)
        # No indicators results in 0.0 (no longer penalized)
        self.assertEqual(out.score, 0.0)


if __name__ == "__main__":
    unittest.main()
