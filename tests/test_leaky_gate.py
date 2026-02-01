"""
Unit tests for Leaky Gate strategy in curriculum learning.

Tests the cold start solution that provides partial rewards during warmup
to help the model learn proper formatting when both format and correctness gates are strict.

Key Policy: NO SHORTCUTS
- Both think and answer tags are REQUIRED from the start
- Missing tags = 0.0 reward, regardless of correctness
- The model cannot shortcut by skipping formatting requirements

Format Validation Requirement:
- format_valid is True ONLY when BOTH think and answer tags are properly formatted
- format_valid is False if either tag is missing or malformed

Leaky Gate Rewards During Warmup:
- Both tags + correct answer: 1.0 (perfect)
- Both tags + wrong answer: 0.1 (partial credit for formatting)
- Missing tags + any answer: 0.0 (NO SHORTCUTS, tags are mandatory)
"""

import unittest
from unittest.mock import MagicMock, patch
from infinite_rl.curriculum import CurriculumLearning, Task, Session
from infinite_rl.reward_functions import RewardFunctionScore


class TestLeakyGate(unittest.TestCase):
    """Test the leaky gate strategy for cold start problem."""

    def setUp(self):
        """Set up test fixtures for leaky gate testing."""
        # Create curriculum with warmup_step=32 (leaky phase lasts 32 steps)
        self.curriculum = CurriculumLearning(
            timeout=10,
            answer_tag="answer",
            think_tag="think",
            warmup_step=32,
            use_format=True,
            aux_weight=0.2,
            num_generations=4,
        )

        # Create a test task
        self.task = Task(
            task_id="test_001",
            task_name="Test Math",
            task_type="math",
            level=0,
            prompt="What is 2 + 2?",
            expected_answer="4",
        )
        self.curriculum.session.add_task(self.task)

    def test_leaky_gate_during_warmup_both_tags_and_answer_correct(self):
        """Test leaky gate returns 1.0 when both tags and answer are correct during warmup."""
        # During warmup (global_step < 32)
        self.assertEqual(self.curriculum.global_step, 0)
        self.assertTrue(self.curriculum.is_warmup())

        # Both tags present and correct answer
        score = self.curriculum._apply_leaky_gate(format_valid=True, primary_score=1.0)
        self.assertEqual(
            score,
            1.0,
            "Should return 1.0 when both tags present and answer correct during warmup",
        )

    def test_leaky_gate_during_warmup_both_tags_only(self):
        """Test leaky gate returns 0.1 when both tags present but answer wrong during warmup."""
        self.assertEqual(self.curriculum.global_step, 0)
        self.assertTrue(self.curriculum.is_warmup())

        # Both tags present, but answer wrong
        score = self.curriculum._apply_leaky_gate(format_valid=True, primary_score=0.0)
        self.assertEqual(
            score,
            0.1,
            "Should return 0.1 when both tags present but answer wrong during warmup",
        )

    def test_leaky_gate_during_warmup_answer_only_no_tags(self):
        """Test leaky gate returns 0.0 (NO SHORTCUTS) when answer correct but tags missing during warmup."""
        self.assertEqual(self.curriculum.global_step, 0)
        self.assertTrue(self.curriculum.is_warmup())

        # Tags missing, but answer correct - NO SHORTCUTS allowed
        score = self.curriculum._apply_leaky_gate(format_valid=False, primary_score=1.0)
        self.assertEqual(
            score,
            0.0,
            "Should return 0.0 (no shortcuts) when answer correct but tags missing during warmup",
        )

    def test_leaky_gate_during_warmup_both_wrong(self):
        """Test leaky gate returns 0.0 when both tags missing and answer wrong during warmup."""
        self.assertEqual(self.curriculum.global_step, 0)
        self.assertTrue(self.curriculum.is_warmup())

        # Both tags missing and answer wrong
        score = self.curriculum._apply_leaky_gate(format_valid=False, primary_score=0.0)
        self.assertEqual(
            score,
            0.0,
            "Should return 0.0 when tags missing and answer wrong during warmup",
        )

    def test_strict_gate_after_warmup_both_correct(self):
        """Test strict gate returns score when both tags and answer correct after warmup."""
        # Simulate being past warmup phase
        self.curriculum.global_step = 32  # At or past warmup_step
        self.assertFalse(self.curriculum.is_warmup())

        # Both tags present and correct answer
        score = self.curriculum._apply_leaky_gate(format_valid=True, primary_score=1.0)
        self.assertEqual(
            score,
            1.0,
            "Should return 1.0 when both tags present and answer correct after warmup",
        )

    def test_strict_gate_after_warmup_format_only(self):
        """Test strict gate returns 0.0 when tags missing after warmup."""
        self.curriculum.global_step = 32
        self.assertFalse(self.curriculum.is_warmup())

        # Tags missing, but answer correct
        score = self.curriculum._apply_leaky_gate(format_valid=False, primary_score=1.0)
        self.assertEqual(
            score,
            0.0,
            "Should return 0.0 when tags missing after warmup (strict gate)",
        )

    def test_strict_gate_after_warmup_answer_only(self):
        """Test strict gate returns 0.0 when answer wrong after warmup."""
        self.curriculum.global_step = 32
        self.assertFalse(self.curriculum.is_warmup())

        # Both tags present, but answer wrong
        score = self.curriculum._apply_leaky_gate(format_valid=True, primary_score=0.0)
        self.assertEqual(
            score,
            0.0,
            "Should return 0.0 when answer wrong after warmup (strict gate)",
        )

    def test_strict_gate_after_warmup_both_wrong(self):
        """Test strict gate returns 0.0 when both tags missing and answer wrong after warmup."""
        self.curriculum.global_step = 32
        self.assertFalse(self.curriculum.is_warmup())

        # Both tags missing and answer wrong
        score = self.curriculum._apply_leaky_gate(format_valid=False, primary_score=0.0)
        self.assertEqual(
            score,
            0.0,
            "Should return 0.0 when tags missing and answer wrong after warmup",
        )

    def test_leaky_gate_with_threshold_score(self):
        """Test leaky gate threshold logic with scores near 0.5."""
        self.curriculum.global_step = 0
        self.assertTrue(self.curriculum.is_warmup())

        # Score exactly at 0.5 threshold (should be treated as incorrect)
        score = self.curriculum._apply_leaky_gate(format_valid=True, primary_score=0.5)
        self.assertEqual(
            score,
            0.1,
            "Score of 0.5 should be treated as incorrect, giving 0.1 with both tags present",
        )

        # Score just above 0.5 (should be treated as correct)
        score = self.curriculum._apply_leaky_gate(format_valid=True, primary_score=0.51)
        self.assertEqual(
            score,
            1.0,
            "Score > 0.5 should be treated as correct with tags during warmup",
        )

    def test_warmup_boundary(self):
        """Test behavior at the exact warmup boundary."""
        # At warmup_step-1, should still be in warmup
        self.curriculum.global_step = 31
        self.assertTrue(self.curriculum.is_warmup())
        score = self.curriculum._apply_leaky_gate(format_valid=True, primary_score=0.0)
        self.assertEqual(
            score, 0.1, "Should still use leaky gate at step 31 with warmup_step=32"
        )

        # At warmup_step, should be past warmup
        self.curriculum.global_step = 32
        self.assertFalse(self.curriculum.is_warmup())
        score = self.curriculum._apply_leaky_gate(format_valid=True, primary_score=0.0)
        self.assertEqual(
            score, 0.0, "Should use strict gate at step 32 with warmup_step=32"
        )

    @patch("infinite_rl.curriculum.CurriculumLearning.get_aux_reward_scores")
    @patch("infinite_rl.curriculum.CurriculumLearning._track_success_group")
    def test_compute_reward_integration_leaky_gate_warmup(self, mock_track, mock_aux):
        """Test that compute_reward properly integrates leaky gate during warmup."""
        # Setup mocks
        mock_aux.return_value = {}
        self.curriculum.grpo_batch_scores = {}
        self.curriculum.grpo_batch_primary_scores = {}
        self.task.model_output = "<think>2+2=4</think><answer>4</answer>"

        # Mock the reward functions to return valid format and correct answer
        with patch.object(
            self.curriculum.aux_reward_functions["format_think"],
            "compute_reward",
            return_value=RewardFunctionScore(
                score=1.0, reward_function_name="format_think", info=""
            ),
        ):
            with patch.object(
                self.curriculum.aux_reward_functions["format_answer"],
                "compute_reward",
                return_value=RewardFunctionScore(
                    score=1.0, reward_function_name="format_answer", info=""
                ),
            ):
                with patch.object(
                    self.curriculum.reward_functions["math"],
                    "compute_reward",
                    return_value=RewardFunctionScore(
                        score=1.0, reward_function_name="primary", info=""
                    ),
                ):
                    # During warmup, correct answer + format
                    score = self.curriculum.compute_reward(
                        "test_001", self.task.model_output
                    )
                    self.assertEqual(
                        score,
                        1.0,
                        "Correct answer with valid format should give 1.0 during warmup",
                    )

    @patch("infinite_rl.curriculum.CurriculumLearning.get_aux_reward_scores")
    @patch("infinite_rl.curriculum.CurriculumLearning._track_success_group")
    def test_compute_reward_leaky_gate_partial_credit(self, mock_track, mock_aux):
        """Test that compute_reward enforces NO SHORTCUTS during leaky gate."""
        # Setup
        mock_aux.return_value = {}
        self.curriculum.grpo_batch_scores = {}
        self.curriculum.grpo_batch_primary_scores = {}
        self.task.model_output = "4"  # Missing XML tags

        # During warmup: correct answer but bad format = NO REWARD (no shortcuts)
        with patch.object(
            self.curriculum.aux_reward_functions["format_think"],
            "compute_reward",
            return_value=RewardFunctionScore(
                score=-1.0, reward_function_name="format_think", info="Missing tags"
            ),
        ):
            with patch.object(
                self.curriculum.reward_functions["math"],
                "compute_reward",
                return_value=RewardFunctionScore(
                    score=1.0, reward_function_name="primary", info=""
                ),
            ):
                score = self.curriculum.compute_reward(
                    "test_001", self.task.model_output
                )
                self.assertEqual(
                    score,
                    0.0,
                    "Should give 0.0 (no shortcuts) for correct answer with missing tags during warmup",
                )

    def test_custom_warmup_step(self):
        """Test that custom warmup_step values work correctly."""
        # Create curriculum with different warmup_step
        curriculum = CurriculumLearning(
            timeout=10,
            warmup_step=100,  # Custom longer warmup
            use_format=True,
        )

        # Leaky gate should be active for 100 steps
        curriculum.global_step = 50
        self.assertTrue(curriculum.is_warmup())
        score = curriculum._apply_leaky_gate(format_valid=True, primary_score=0.0)
        self.assertEqual(
            score, 0.1, "Leaky gate should be active at step 50 with warmup_step=100"
        )

        # At step 100, should be past warmup
        curriculum.global_step = 100
        self.assertFalse(curriculum.is_warmup())
        score = curriculum._apply_leaky_gate(format_valid=True, primary_score=0.0)
        self.assertEqual(
            score, 0.0, "Strict gate should be active at step 100 with warmup_step=100"
        )


class TestLeakyGateEdgeCases(unittest.TestCase):
    """Test edge cases and boundary conditions for leaky gate."""

    def setUp(self):
        """Set up test fixtures."""
        self.curriculum = CurriculumLearning(
            timeout=10,
            warmup_step=10,
            use_format=True,
        )

    def test_partial_correctness_scores(self):
        """Test leaky gate with partial correctness scores between 0 and 1."""
        self.curriculum.global_step = 0

        # Score of 0.6 (just above threshold)
        score = self.curriculum._apply_leaky_gate(format_valid=True, primary_score=0.6)
        self.assertEqual(
            score, 1.0, "Score > 0.5 should give full credit during warmup"
        )

        # Score of 0.4 (just below threshold)
        score = self.curriculum._apply_leaky_gate(format_valid=True, primary_score=0.4)
        self.assertEqual(
            score, 0.1, "Score < 0.5 should give partial credit during warmup"
        )

    def test_zero_warmup_step(self):
        """Test behavior when warmup_step is 0 (no leaky phase)."""
        curriculum = CurriculumLearning(
            timeout=10,
            warmup_step=0,
            use_format=True,
        )

        # From step 0, should be past warmup
        curriculum.global_step = 0
        self.assertFalse(curriculum.is_warmup())

        # Should enforce strict gate immediately
        score = curriculum._apply_leaky_gate(format_valid=True, primary_score=0.0)
        self.assertEqual(score, 0.0, "Should enforce strict gate when warmup_step=0")

    def test_very_large_warmup_step(self):
        """Test behavior with very large warmup_step."""
        curriculum = CurriculumLearning(
            timeout=10,
            warmup_step=1000000,
            use_format=True,
        )

        # Very far into training, still in warmup
        curriculum.global_step = 999999
        self.assertTrue(curriculum.is_warmup())

        score = curriculum._apply_leaky_gate(format_valid=True, primary_score=0.0)
        self.assertEqual(score, 0.1, "Should still use leaky gate within warmup phase")

        # Just past warmup
        curriculum.global_step = 1000000
        self.assertFalse(curriculum.is_warmup())

        score = curriculum._apply_leaky_gate(format_valid=True, primary_score=0.0)
        self.assertEqual(score, 0.0, "Should switch to strict gate after warmup")


if __name__ == "__main__":
    unittest.main()
