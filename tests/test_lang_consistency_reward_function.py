import unittest
from pathlib import Path
from infinite_rl.reward_functions.lang_consistency import LangConsistencyRewardFunction
from infinite_rl.task import Task


class TestLangConsistencyRewardFunction(unittest.TestCase):
    """Test language/dialect reward function renamed to lang_consistency."""

    def setUp(self):
        self.reward_fn = LangConsistencyRewardFunction(task_name="lang_consistency")
        self.reward_fn.initialize()

    def test_language_mismatch(self):
        # Expected Cantonese, but response is Mandarin/Chinese inside <answer> ->
        task = Task(
            task_id="test_1",
            task_name="test",
            task_type="math",
            level=1,
            prompt="Test",
            expected_answer="",
            language="yue",
            model_output="<answer>这是普通话。</answer>",
        )
        score = self.reward_fn.compute_reward(task)
        self.assertAlmostEqual(score.score, 0.25, places=3)

    def test_language_match(self):
        task = Task(
            task_id="test_2",
            task_name="test",
            task_type="math",
            level=1,
            prompt="Test",
            expected_answer="",
            language="yue",
            model_output="<answer>我愛你。</answer>",
        )
        score = self.reward_fn.compute_reward(task)
        self.assertAlmostEqual(score.score, 1.0, places=3)
