"""Unit tests for num_generations parameter in CurriculumLearning."""

import unittest
from unittest.mock import MagicMock, patch
from infinite_rl.curriculum import CurriculumLearning


class TestNumGenerations(unittest.TestCase):
    """Test num_generations parameter initialization and functionality."""

    def test_num_generations_default(self):
        """Test that num_generations defaults to 4."""
        curriculum = CurriculumLearning()
        self.assertEqual(curriculum.num_generations, 4)

    def test_num_generations_custom_value(self):
        """Test that num_generations can be set to custom values."""
        for num_gen in [1, 2, 4, 8, 16]:
            curriculum = CurriculumLearning(num_generations=num_gen)
            self.assertEqual(curriculum.num_generations, num_gen)

    def test_num_generations_batch_accumulation(self):
        """Test that GRPO batch accumulates up to num_generations."""
        curriculum = CurriculumLearning(num_generations=2)

        # Add a task to the session
        curriculum.get_prompt()
        task_id = list(curriculum.session.tasks.keys())[0]
        base_task_id = task_id.rsplit("_", 1)[0] if "_" in task_id else task_id

        # First response - batch should accumulate
        curriculum.compute_reward(task_id, "response 1")
        self.assertIn(base_task_id, curriculum.grpo_batch_scores)
        self.assertEqual(len(curriculum.grpo_batch_scores[base_task_id]), 1)

        # Second response - batch should still have 1 (previous was cleared after it hit num_generations)
        # Actually, we need to create a second task instance
        curriculum.get_prompt()
        task_id_2 = list(curriculum.session.tasks.keys())[-1]
        base_task_id_2 = task_id_2.rsplit("_", 1)[0] if "_" in task_id_2 else task_id_2

        if base_task_id_2 == base_task_id:
            curriculum.compute_reward(task_id_2, "response 2")
            # Should have triggered level update since we reached num_generations
            self.assertNotIn(base_task_id, curriculum.grpo_batch_scores)

    def test_num_generations_level_update_trigger(self):
        """Test that level update is triggered when num_generations responses are accumulated."""
        curriculum = CurriculumLearning(
            num_generations=2,
            window_size=10,
            warmup_step=0,  # Disable warmup for this test
        )

        # Get first prompt
        task1 = curriculum.get_prompt()
        self.assertIsNotNone(task1)

        # Get second prompt (same base task due to instance counter)
        task2 = curriculum.get_prompt()
        self.assertIsNotNone(task2)

        # Both should have different instance task_ids
        self.assertNotEqual(task1.task_id, task2.task_id)

        # Submit responses for both
        initial_step = curriculum.global_step
        curriculum.compute_reward(task1.task_id, "test response 1")
        curriculum.compute_reward(task2.task_id, "test response 2")

        # global_step should have incremented by 2
        self.assertEqual(curriculum.global_step, initial_step + 2)

    def test_num_generations_one(self):
        """Test behavior with num_generations=1 (immediate level updates)."""
        curriculum = CurriculumLearning(
            num_generations=1, warmup_step=0  # Disable warmup
        )

        task = curriculum.get_prompt()
        self.assertIsNotNone(task)

        # With num_generations=1, a single response should trigger level update
        # (batch should complete immediately)
        curriculum.compute_reward(task.task_id, "response")

        # Verify batch was cleaned up
        base_task_id = (
            task.task_id.rsplit("_", 1)[0] if "_" in task.task_id else task.task_id
        )
        self.assertNotIn(base_task_id, curriculum.grpo_batch_scores)

    def test_num_generations_with_different_initialization(self):
        """Test initialization with other parameters and num_generations."""
        curriculum = CurriculumLearning(
            timeout=20,
            answer_tag="ans",
            num_generations=8,
            window_size=100,
            success_rate_threshold=0.75,
        )

        self.assertEqual(curriculum.num_generations, 8)
        self.assertEqual(curriculum.timeout, 20)
        self.assertEqual(curriculum.answer_tag, "ans")
        self.assertEqual(curriculum.window_size, 100)
        self.assertEqual(curriculum.success_rate_threshold, 0.75)


if __name__ == "__main__":
    unittest.main()
