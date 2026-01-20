import unittest
from pathlib import Path
from infinite_rl.reward_functions.lang_consistency import LangConsistencyRewardFunction


class TestLangConsistencyRewardFunction(unittest.TestCase):
    """Test language/dialect reward function renamed to lang_consistency."""

    def setUp(self):
        self.reward_fn = LangConsistencyRewardFunction(task_name="lang_consistency")
        self.reward_fn.initialize()

    def test_language_mismatch(self):
        # Expected Cantonese, but response is Mandarin/Chinese inside <answer> ->
        score = self.reward_fn.compute_reward("这是普通话。", "yue")
        self.assertAlmostEqual(score.score, 0.25, places=3)

    def test_language_match(self):
        score = self.reward_fn.compute_reward("我愛你。", "yue")
        self.assertAlmostEqual(score.score, 1.0, places=3)
