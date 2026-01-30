import unittest
from pathlib import Path
from infinite_rl.reward_functions.lang_consistency import LangConsistencyRewardFunction
from infinite_rl.task import Task


class TestLangConsistencyRewardFunction(unittest.TestCase):
    """Test language/dialect reward function renamed to lang_consistency."""

    def setUp(self):
        self.reward_fn = LangConsistencyRewardFunction(
            task_name="lang_consistency", tag_excluded=False
        )
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
        self.assertEqual(score.score, -4.0)

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

    def test_answer_tag_included_default(self):
        """Test that by default (tag_excluded=False), content inside <answer> tag is checked."""
        reward_fn = LangConsistencyRewardFunction(
            task_name="lang_consistency", tag_excluded=False
        )
        reward_fn.initialize()

        # Response has English outside <answer> but Cantonese inside
        # Should evaluate the Cantonese inside <answer> tag
        task = Task(
            task_id="test_3",
            task_name="test",
            task_type="math",
            level=1,
            prompt="Test",
            expected_answer="",
            language="yue",
            model_output="Some English text outside. <answer>我愛你。</answer>",
        )
        score = reward_fn.compute_reward(task)
        # Should detect Cantonese from inside the <answer> tag
        self.assertAlmostEqual(score.score, 1.0, places=3)

    def test_tag_excluded_true(self):
        """Test that with tag_excluded=True, content OUTSIDE <answer> tag is checked."""
        reward_fn = LangConsistencyRewardFunction(
            task_name="lang_consistency", tag_excluded=True
        )
        reward_fn.initialize()

        # Response has Cantonese outside <answer> but English inside
        # Should evaluate the Cantonese outside <answer> tag
        task = Task(
            task_id="test_4",
            task_name="test",
            task_type="math",
            level=1,
            prompt="Test",
            expected_answer="",
            language="yue",
            model_output="我愛你。<answer>This is English</answer>",
        )
        score = reward_fn.compute_reward(task)
        # Should detect Cantonese from outside the <answer> tag
        self.assertAlmostEqual(score.score, 1.0, places=3)

    def test_tag_excluded_with_prefix_and_suffix(self):
        """Test tag_excluded=True with both prefix and suffix outside the tag."""
        reward_fn = LangConsistencyRewardFunction(
            task_name="lang_consistency", tag_excluded=True
        )
        reward_fn.initialize()

        task = Task(
            task_id="test_5",
            task_name="test",
            task_type="math",
            level=1,
            prompt="Test",
            expected_answer="",
            language="yue",
            model_output="前言：我愛你 <answer>English content here</answer> 後言：多謝你。",
        )
        score = reward_fn.compute_reward(task)
        # Should detect Cantonese from both prefix and suffix (outside tags)
        self.assertAlmostEqual(score.score, 1.0, places=3)

    def test_tag_excluded_mismatch(self):
        """Test tag_excluded=True with mismatched language outside tag."""
        reward_fn = LangConsistencyRewardFunction(
            task_name="lang_consistency", tag_excluded=True
        )
        reward_fn.initialize()

        task = Task(
            task_id="test_6",
            task_name="test",
            task_type="math",
            level=1,
            prompt="Test",
            expected_answer="",
            language="yue",
            # Substantial Mandarin text outside the tag to ensure clear mismatch
            model_output="这是普通话 这是普通话 这是普通话 <answer>我愛你。</answer>",
        )
        score = reward_fn.compute_reward(task)
        # Should detect Mandarin/SWC from outside, not Cantonese inside
        # Since expected is Cantonese but detected is Mandarin, score should be -4.0
        self.assertEqual(score.score, -4.0)
