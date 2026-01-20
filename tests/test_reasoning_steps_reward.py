import unittest
from infinite_rl.reward_functions.reasoning_steps import ReasoningStepsRewardFunction


class TestReasoningStepsRewardFunction(unittest.TestCase):
    def setUp(self):
        self.rf = ReasoningStepsRewardFunction()
        self.rf.initialize()

    def test_missing_think_tag(self):
        out = self.rf.compute_reward("I think the answer is 2.", None)
        self.assertEqual(out.score, 0.0)

    def test_single_indicator(self):
        model_output = "<think>First, we compute the sum.</think>"
        out = self.rf.compute_reward(model_output, None)
        # Aux-only: returned in unified score field
        self.assertAlmostEqual(out.score, 0.1, places=5)

    def test_multiple_unique_indicators(self):
        model_output = "<think>First, we compute. Second, we verify. Finally, we present the result.</think>"
        out = self.rf.compute_reward(model_output, None)
        self.assertAlmostEqual(out.score, 0.2, places=5)

    def test_repeated_indicators_count_once(self):
        model_output = "<think>First. First. First.</think>"
        out = self.rf.compute_reward(model_output, None)
        self.assertAlmostEqual(out.score, 0.1, places=5)


if __name__ == "__main__":
    unittest.main()
