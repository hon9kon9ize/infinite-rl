import unittest
from infinite_rl.reward_functions.length import (
    LengthRewardFunction,
    reasoning_friendly_length_reward,
)
from infinite_rl.task import Task


class TestLengthRewardFunction(unittest.TestCase):
    def setUp(self):
        self.fn = LengthRewardFunction(task_name="length", target_tag="think")

    def test_reasoning_friendly_length_reward_too_short(self):
        """Test that very short responses get low scores."""
        # Length < 100 should return 0.1
        score = reasoning_friendly_length_reward(50, 1300)
        self.assertEqual(score, 0.1)

        score = reasoning_friendly_length_reward(99, 1300)
        self.assertEqual(score, 0.1)

    def test_reasoning_friendly_length_reward_sweet_spot(self):
        """Test that responses in the sweet spot get perfect scores."""
        # Length between 100 and target_len should return 1.0
        score = reasoning_friendly_length_reward(100, 1300)
        self.assertEqual(score, 1.0)

        score = reasoning_friendly_length_reward(800, 1300)
        self.assertEqual(score, 1.0)

        score = reasoning_friendly_length_reward(1299, 1300)
        self.assertEqual(score, 1.0)

    def test_reasoning_friendly_length_reward_decay(self):
        """Test that overly long responses get decaying scores."""
        # Length > target_len should decay towards 0.5
        score = reasoning_friendly_length_reward(1301, 1300, 3584)
        self.assertLess(score, 1.0)
        self.assertGreater(score, 0.5)

        # At max_len, should be 0.5
        score = reasoning_friendly_length_reward(3584, 1300, 3584)
        self.assertEqual(score, 0.5)

    def test_reasoning_friendly_length_reward_edge_cases(self):
        """Test edge cases for the reasoning_friendly_length_reward function."""
        # Target length >= max_len should return 0.5 for anything over target
        score = reasoning_friendly_length_reward(
            2100, 2000, 1500
        )  # target_len > max_len, length > target_len
        self.assertEqual(score, 0.5)

    def test_level_specific_target_lengths(self):
        """Test that different task levels use correct target lengths."""
        # Level 0: 1300
        task0 = Task(
            task_id="test_1",
            task_name="test",
            task_type="math",
            level=0,
            prompt="Test",
            expected_answer="",
        )
        task0.model_output = "<think>" + "x" * 2000 + "</think><answer>42</answer>"
        score = self.fn.compute_reward(task0, is_correct=True)
        # Should be ~0.89 since length is in decay region for level 0 (2000 > 1300)
        self.assertAlmostEqual(score.score, 0.89, delta=0.01)

        # Level 1: 1600
        task1 = Task(
            task_id="test_2",
            task_name="test",
            task_type="math",
            level=1,
            prompt="Test",
            expected_answer="",
        )
        task1.model_output = "<think>" + "x" * 2000 + "</think><answer>42</answer>"
        score = self.fn.compute_reward(task1, is_correct=True)
        self.assertAlmostEqual(score.score, 0.95, delta=0.01)  # 2000 > 1600

        # Level 2: 2000
        task2 = Task(
            task_id="test_3",
            task_name="test",
            task_type="math",
            level=2,
            prompt="Test",
            expected_answer="",
        )
        task2.model_output = "<think>" + "x" * 2000 + "</think><answer>42</answer>"
        score = self.fn.compute_reward(task2, is_correct=True)
        self.assertEqual(score.score, 1.0)  # 2000 = 2000

        # Level 3+: 3000
        task3 = Task(
            task_id="test_4",
            task_name="test",
            task_type="math",
            level=3,
            prompt="Test",
            expected_answer="",
        )
        task3.model_output = "<think>" + "x" * 2000 + "</think><answer>42</answer>"
        score = self.fn.compute_reward(task3, is_correct=True)
        self.assertEqual(score.score, 1.0)  # 2000 < 3000

    def test_missing_think_tag(self):
        """Test behavior when think tag is missing."""
        task = Task(
            task_id="test_2",
            task_name="test",
            task_type="math",
            level=0,
            prompt="Test",
            expected_answer="",
            model_output="<answer>42</answer>",  # No think tag
        )
        score = self.fn.compute_reward(task)
        self.assertEqual(score.score, 0.0)
        self.assertIn("No content found in the <think> tag", score.info)

    def test_empty_think_content(self):
        """Test behavior with empty think content."""
        task = Task(
            task_id="test_3",
            task_name="test",
            task_type="math",
            level=0,
            prompt="Test",
            expected_answer="",
            model_output="<think></think><answer>42</answer>",
        )
        score = self.fn.compute_reward(task)
        self.assertEqual(score.score, 0.0)
        self.assertIn("No content found in the <think> tag", score.info)

    def test_whitespace_only_content(self):
        """Test behavior with whitespace-only think content."""
        task = Task(
            task_id="test_4",
            task_name="test",
            task_type="math",
            level=0,
            prompt="Test",
            expected_answer="",
            model_output="<think>   \n\t   </think><answer>42</answer>",
        )
        score = self.fn.compute_reward(task)
        self.assertEqual(score.score, 0.0)
        self.assertIn("No content found in the <think> tag", score.info)

    def test_too_short_response(self):
        """Test that very short responses get penalized."""
        task = Task(
            task_id="test_5",
            task_name="test",
            task_type="math",
            level=0,
            prompt="Test",
            expected_answer="",
            model_output="<think>Hi</think><answer>42</answer>",
        )
        score = self.fn.compute_reward(task, is_correct=True)
        self.assertEqual(score.score, 0.1)  # Too short

    def test_too_long_response(self):
        """Test that overly long responses get decaying scores."""
        task = Task(
            task_id="test_6",
            task_name="test",
            task_type="math",
            level=0,
            prompt="Test",
            expected_answer="",
            model_output="<think>"
            + "x" * 2000
            + "</think><answer>42</answer>",  # Much longer than level 0 target
        )
        score = self.fn.compute_reward(task, is_correct=True)
        self.assertLess(score.score, 1.0)
        self.assertGreater(score.score, 0.5)

    def test_custom_target_tag(self):
        """Test using a custom target tag."""
        fn = LengthRewardFunction(task_name="length", target_tag="reasoning")
        fn.initialize()

        task = Task(
            task_id="test_7",
            task_name="test",
            task_type="math",
            level=0,
            prompt="Test",
            expected_answer="",
            model_output="<reasoning>"
            + "This is reasonable length content that should be long enough for a good score. "
            * 10
            + "</reasoning><answer>42</answer>",
        )
        score = fn.compute_reward(task, is_correct=True)
        self.assertEqual(score.score, 1.0)

    def test_custom_max_len(self):
        """Test custom max_len parameter."""
        fn = LengthRewardFunction(task_name="length", max_len=2000)
        fn.initialize()

        task = Task(
            task_id="test_8",
            task_name="test",
            task_type="math",
            level=0,
            prompt="Test",
            expected_answer="",
            model_output="<think>" + "x" * 1800 + "</think><answer>42</answer>",
        )
        score = fn.compute_reward(task, is_correct=True)
        # With max_len=2000 and target=1300, this should be in decay region
        self.assertLess(score.score, 1.0)
        self.assertGreater(score.score, 0.5)

    def test_unknown_level_defaults(self):
        """Test that unknown levels default to target_len=2000."""
        task = Task(
            task_id="test_9",
            task_name="test",
            task_type="math",
            level=99,  # Unknown level
            prompt="Test",
            expected_answer="",
            model_output="<think>" + "x" * 1800 + "</think><answer>42</answer>",
        )
        score = self.fn.compute_reward(task, is_correct=True)
        # Should use default target_len=2000, so this is in sweet spot
        self.assertEqual(score.score, 1.0)

    def test_correctness_gating(self):
        """Test that length reward is gated by correctness."""
        task = Task(
            task_id="test_10",
            task_name="test",
            task_type="math",
            level=0,
            prompt="Test",
            expected_answer="",
            model_output="<think>"
            + "x" * 1500
            + "</think><answer>42</answer>",  # Good length
        )

        # When correct, should get normal length reward
        score_correct = self.fn.compute_reward(task, is_correct=True)
        self.assertGreater(score_correct.score, 0.8)  # Should be in sweet spot

        # When incorrect, should get 0.0 regardless of length
        score_incorrect = self.fn.compute_reward(task, is_correct=False)
        self.assertEqual(score_incorrect.score, 0.0)


if __name__ == "__main__":
    unittest.main()
