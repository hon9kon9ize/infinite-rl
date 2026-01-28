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
        # Aux-only: returned in unified score field
        self.assertAlmostEqual(out.score, 0.1, places=5)

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
        self.assertAlmostEqual(out.score, 0.2, places=5)

    def test_repeated_indicators_count_once(self):
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
        self.assertAlmostEqual(out.score, 0.1, places=5)


if __name__ == "__main__":
    unittest.main()
