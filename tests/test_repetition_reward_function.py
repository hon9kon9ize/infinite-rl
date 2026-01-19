import unittest
from infinite_rl.reward_functions.repetition import RepetitionRewardFunction


class TestRepetitionRewardFunction(unittest.TestCase):
    def setUp(self):
        self.fn = RepetitionRewardFunction()
        self.fn.initialize()

    def test_no_repetition_full_score(self):
        s = self.fn.compute_reward("<answer>hello world</answer>", None)
        # Aux-only reward: format and correctness are zero, aux_score carries the signal
        self.assertEqual(s.format_score, 0.0)
        self.assertEqual(s.correctness_score, 0.0)
        self.assertAlmostEqual(s.aux_score, 1.0, places=5)

    def test_repetition_penalized(self):
        s = self.fn.compute_reward("<answer>hello hello hello hello</answer>", None)
        self.assertEqual(s.correctness_score, 0.0)
        self.assertLess(s.aux_score, 1.0)


if __name__ == "__main__":
    unittest.main()
