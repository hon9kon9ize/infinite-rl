import unittest
from infinite_rl import RewardOrchestrator
from infinite_rl.reward_functions.reward_function import RewardFunctionScore


class TestRewardOrchestrator(unittest.TestCase):
    def setUp(self):
        self.orch = RewardOrchestrator(timeout=2)

    def test_available_contains_math(self):
        av = self.orch.available()
        self.assertIn("math", av)
        self.assertIn("coding", av)

    def test_compute_math(self):
        example_out = "<answer>42</answer>"
        score = self.orch.compute(example_out, "42", task="math")
        self.assertEqual(score.format_score, 1.0)
        self.assertEqual(score.correctness_score, 1.0)

    def test_repetition_wrapper(self):
        orch = RewardOrchestrator(include_repetition=True)
        self.assertIn("repetition", orch.available())
        # Low repetition (aux-only behavior)
        s1 = orch.compute("<answer>Hello world</answer>", None, task="repetition")
        self.assertEqual(s1.format_score, 0.0)
        self.assertEqual(s1.correctness_score, 0.0)
        self.assertAlmostEqual(s1.aux_score, 1.0, places=3)
        # High repetition
        s2 = orch.compute(
            "<answer>Hello Hello Hello Hello</answer>", None, task="repetition"
        )
        self.assertEqual(s2.correctness_score, 0.0)
        self.assertLess(s2.aux_score, s1.aux_score)

    def test_compute_with_lang_and_length(self):
        orch = RewardOrchestrator(include_length=True)
        # main task: coding (exists), auxiliary: lang=en
        res = orch.compute(
            "<answer>Hello world</answer>",
            "",
            task="coding",
            lang="en",
        )
        # New API: compute returns a single aggregated RewardFunctionScore
        self.assertIsInstance(res, RewardFunctionScore)
        # main reward unchanged (coding correctness may be 0 unless code block exists)
        self.assertEqual(res.format_score, 0.5)
        self.assertEqual(res.correctness_score, 0.0)
        # aux_score should be the sum of lang and length signals
        self.assertIsInstance(res.aux_score, float)

    def test_length_wrapper(self):
        orch = RewardOrchestrator(include_length=True, length_target_len=10)
        self.assertIn("length", orch.available())
        # Short correct (simulate expected == content)
        s_short = orch.compute("<answer>short</answer>", "short", task="length")
        # Aux-only length reward: format/correctness zero, signal is in aux_score
        self.assertEqual(s_short.format_score, 0.0)
        self.assertEqual(s_short.correctness_score, 0.0)
        # Longer correct should be <= short (compare aux_score)
        long_text = "word " * 50
        s_long = orch.compute(
            f"<answer>{long_text}</answer>", long_text.strip(), task="length"
        )
        self.assertLessEqual(s_long.aux_score, s_short.aux_score)

    def test_length_respects_main_correctness(self):
        orch = RewardOrchestrator(include_length=True)
        length_fn = orch.get_fn("length")

        # When main task is correct, shorter answers should get higher aux_score than longer ones
        len_short_corr = length_fn.compute_reward(
            "<answer>3</answer>", 1, is_correct=True
        )
        len_long_corr = length_fn.compute_reward(
            f"<answer>{'word '*50}</answer>", 1, is_correct=True
        )
        self.assertGreater(len_short_corr.aux_score, len_long_corr.aux_score)

        # When main task is incorrect, longer answers should get higher aux_score compared to short answers
        len_short_inc = length_fn.compute_reward(
            "<answer>short</answer>", 1, is_correct=False
        )
        len_long_inc = length_fn.compute_reward(
            f"<answer>{'word '*50}</answer>", 1, is_correct=False
        )
        self.assertGreater(len_long_inc.aux_score, len_short_inc.aux_score)


if __name__ == "__main__":
    unittest.main()
