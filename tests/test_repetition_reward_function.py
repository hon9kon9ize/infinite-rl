import unittest
from infinite_rl.reward_functions.repetition import RepetitionRewardFunction


class TestRepetitionRewardFunction(unittest.TestCase):
    def setUp(self):
        self.fn = RepetitionRewardFunction()
        self.fn.initialize()

    def test_no_repetition_full_score(self):
        s = self.fn.compute_reward("<answer>hello world</answer>", None)
        # Aux-only reward: returned in unified `score` field
        self.assertAlmostEqual(s.score, 1.0, places=5)

    def test_repetition_penalized(self):
        s = self.fn.compute_reward("<answer>hello hello hello hello</answer>", None)
        # High repetition -> lower score than 1.0
        self.assertLess(s.score, 1.0)
        self.assertGreaterEqual(s.score, 0.0)


if __name__ == "__main__":
    unittest.main()
