"""Unit tests for num_generations parameter and GRPO batching in CurriculumLearning.

This module tests the critical GRPO batching fix that ensures:
1. Multiple responses for the same prompt (same base_task_id) are grouped correctly
2. global_step increments only once per complete batch (not per response)
3. Level updates only happen when batches are complete
4. Stale incomplete batches are cleaned up to prevent memory leaks
5. Oversized batches are handled gracefully

These tests verify the fix for the issue where DynamicCurriculumDataset was
creating different tasks for each GRPO response instead of reusing the same task.

Note: Integration tests for DynamicCurriculumDataset are in the main training script
      as it requires complex mocking of transformers, wandb, and other dependencies.
"""

import unittest
from unittest.mock import MagicMock, patch
from infinite_rl.curriculum import CurriculumLearning, Task


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

        # Get one prompt
        task = curriculum.get_prompt()
        self.assertIsNotNone(task)
        base_task_id = task.task_id.rsplit("_", 1)[0]

        # Submit 2 responses for the same task (complete batch for num_generations=2)
        initial_step = curriculum.global_step
        curriculum.compute_reward(task.task_id, "test response 1")
        # Still incomplete batch - step shouldn't increment yet
        self.assertEqual(curriculum.global_step, initial_step)

        # Add second response with same base_task_id (simulating GRPO)
        task_id_2 = f"{base_task_id}_1"
        # Need to add this task to session for compute_reward to work
        task2 = Task(
            task_id=task_id_2,
            task_name=task.task_name,
            task_type=task.task_type,
            level=task.level,
            prompt=task.prompt,
            expected_answer=task.expected_answer,
        )
        curriculum.session.add_task(task2)

        curriculum.compute_reward(task_id_2, "test response 2")

        # global_step should have incremented by 1 (once per complete batch)
        self.assertEqual(curriculum.global_step, initial_step + 1)

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


class TestGRPOBatching(unittest.TestCase):
    """Test GRPO batching behavior with proper task ID reuse."""

    def setUp(self):
        """Set up test fixtures."""
        self.curriculum = CurriculumLearning(
            num_generations=4,
            warmup_step=0,  # Disable warmup for clearer testing
            use_format=False,
            use_repetition=False,
            use_reasoning_steps=False,
            use_length=False,
            use_lang_consistency=False,
        )

    def test_grpo_batch_accumulation_with_same_base_task_id(self):
        """Test that multiple responses with same base task_id are grouped correctly."""
        # Create tasks for all 4 responses (simulating how DynamicCurriculumDataset works)
        base_task_id = "math_test_123"
        for i in range(4):
            task = Task(
                task_id=f"{base_task_id}_{i}",
                task_name="Test Math",
                task_type="math",
                level=1,
                prompt="What is 2 + 2?",
                expected_answer="4",
            )
            self.curriculum.session.add_task(task)

        # Mock reward function to return success
        with patch.object(
            self.curriculum.reward_functions["math"], "compute_reward"
        ) as mock_compute:
            mock_result = MagicMock()
            mock_result.score = 1.0
            mock_result.info = ""
            mock_compute.return_value = mock_result

            initial_step = self.curriculum.global_step

            # Response 1
            self.curriculum.compute_reward(f"{base_task_id}_0", "<answer>4</answer>")
            self.assertIn(base_task_id, self.curriculum.grpo_batch_scores)
            self.assertEqual(len(self.curriculum.grpo_batch_scores[base_task_id]), 1)
            self.assertEqual(
                self.curriculum.global_step, initial_step
            )  # Not incremented yet

            # Response 2
            self.curriculum.compute_reward(f"{base_task_id}_1", "<answer>4</answer>")
            self.assertEqual(len(self.curriculum.grpo_batch_scores[base_task_id]), 2)
            self.assertEqual(
                self.curriculum.global_step, initial_step
            )  # Still not incremented

            # Response 3
            self.curriculum.compute_reward(f"{base_task_id}_2", "<answer>4</answer>")
            self.assertEqual(len(self.curriculum.grpo_batch_scores[base_task_id]), 3)
            self.assertEqual(
                self.curriculum.global_step, initial_step
            )  # Still not incremented

            # Response 4 - should trigger batch completion
            self.curriculum.compute_reward(f"{base_task_id}_3", "<answer>4</answer>")

            # Batch should be cleared
            self.assertNotIn(base_task_id, self.curriculum.grpo_batch_scores)

            # global_step should increment ONCE for the entire batch
            self.assertEqual(self.curriculum.global_step, initial_step + 1)

    def test_grpo_batch_prevents_premature_level_update(self):
        """Test that level updates only happen when batch is complete."""
        curriculum = CurriculumLearning(
            num_generations=4,
            warmup_step=0,
            window_size=10,
            success_rate_threshold=0.8,
            use_format=False,
            use_repetition=False,
            use_reasoning_steps=False,
            use_length=False,
            use_lang_consistency=False,
        )

        # Create tasks for all 4 responses
        base_task_id = "math_test_456"
        for i in range(4):
            task = Task(
                task_id=f"{base_task_id}_{i}",
                task_name="Test Math",
                task_type="math",
                level=1,
                prompt="What is 2 + 2?",
                expected_answer="4",
            )
            curriculum.session.add_task(task)

        with patch.object(
            curriculum.reward_functions["math"], "compute_reward"
        ) as mock_compute:
            mock_result = MagicMock()
            mock_result.score = 1.0
            mock_result.info = ""
            mock_compute.return_value = mock_result

            initial_level = curriculum.current_level

            # Submit 3 responses (incomplete batch)
            for i in range(3):
                curriculum.compute_reward(f"{base_task_id}_{i}", "<answer>4</answer>")

            # Level should NOT change with incomplete batch
            self.assertEqual(curriculum.current_level, initial_level)
            self.assertIn(base_task_id, curriculum.grpo_batch_scores)

            # Complete the batch
            curriculum.compute_reward(f"{base_task_id}_3", "<answer>4</answer>")

            # Now the batch is complete and should be processed
            self.assertNotIn(base_task_id, curriculum.grpo_batch_scores)

    def test_grpo_batch_handles_oversized_batch(self):
        """Test that oversized batches (more responses than expected) are handled."""
        base_task_id = "math_test_789"
        for i in range(5):  # 5 responses (oversized)
            task = Task(
                task_id=f"{base_task_id}_{i}",
                task_name="Test Math",
                task_type="math",
                level=1,
                prompt="What is 2 + 2?",
                expected_answer="4",
            )
            self.curriculum.session.add_task(task)

        with patch.object(
            self.curriculum.reward_functions["math"], "compute_reward"
        ) as mock_compute:
            mock_result = MagicMock()
            mock_result.score = 1.0
            mock_result.info = ""
            mock_compute.return_value = mock_result

            initial_step = self.curriculum.global_step

            # Submit 5 responses (1 more than num_generations=4)
            for i in range(5):
                self.curriculum.compute_reward(
                    f"{base_task_id}_{i}", "<answer>4</answer>"
                )

            # Should have processed the first 4 and cleared the batch
            # The 5th response would start a new batch (with 1 score in it)
            self.assertIn(base_task_id, self.curriculum.grpo_batch_scores)
            self.assertEqual(len(self.curriculum.grpo_batch_scores[base_task_id]), 1)
            self.assertEqual(self.curriculum.global_step, initial_step + 1)

    def test_grpo_batch_different_tasks_independent(self):
        """Test that different tasks maintain independent batch counters."""
        # Create two sets of tasks
        base_task_id1 = "math_test_111"
        base_task_id2 = "math_test_222"

        for i in range(4):
            task1 = Task(
                task_id=f"{base_task_id1}_{i}",
                task_name="Test Math 1",
                task_type="math",
                level=1,
                prompt="What is 2 + 2?",
                expected_answer="4",
            )
            task2 = Task(
                task_id=f"{base_task_id2}_{i}",
                task_name="Test Math 2",
                task_type="math",
                level=1,
                prompt="What is 3 + 3?",
                expected_answer="6",
            )
            self.curriculum.session.add_task(task1)
            self.curriculum.session.add_task(task2)

        with patch.object(
            self.curriculum.reward_functions["math"], "compute_reward"
        ) as mock_compute:
            mock_result = MagicMock()
            mock_result.score = 1.0
            mock_result.info = ""
            mock_compute.return_value = mock_result

            # Submit 2 responses for task1
            self.curriculum.compute_reward(f"{base_task_id1}_0", "<answer>4</answer>")
            self.curriculum.compute_reward(f"{base_task_id1}_1", "<answer>4</answer>")

            # Submit 3 responses for task2
            self.curriculum.compute_reward(f"{base_task_id2}_0", "<answer>6</answer>")
            self.curriculum.compute_reward(f"{base_task_id2}_1", "<answer>6</answer>")
            self.curriculum.compute_reward(f"{base_task_id2}_2", "<answer>6</answer>")

            # Both should have independent counters
            self.assertIn(base_task_id1, self.curriculum.grpo_batch_scores)
            self.assertIn(base_task_id2, self.curriculum.grpo_batch_scores)
            self.assertEqual(len(self.curriculum.grpo_batch_scores[base_task_id1]), 2)
            self.assertEqual(len(self.curriculum.grpo_batch_scores[base_task_id2]), 3)

    def test_grpo_batch_cleanup_stale_batches(self):
        """Test that stale incomplete batches are cleaned up periodically."""
        curriculum = CurriculumLearning(
            num_generations=4,
            warmup_step=0,
            use_format=False,
            use_repetition=False,
            use_reasoning_steps=False,
            use_length=False,
            use_lang_consistency=False,
        )

        # Artificially create many incomplete batches
        for i in range(60):
            curriculum.grpo_batch_scores[f"stale_task_{i}"] = [0.5]

        # Create tasks for all 4 responses
        base_task_id = "math_test_999"
        for i in range(4):
            task = Task(
                task_id=f"{base_task_id}_{i}",
                task_name="Test Math",
                task_type="math",
                level=1,
                prompt="What is 2 + 2?",
                expected_answer="4",
            )
            curriculum.session.add_task(task)

        with patch.object(
            curriculum.reward_functions["math"], "compute_reward"
        ) as mock_compute:
            mock_result = MagicMock()
            mock_result.score = 1.0
            mock_result.info = ""
            mock_compute.return_value = mock_result

            # Force global_step to trigger cleanup
            curriculum.global_step = 99  # Next step will be 100

            # Complete a batch to increment global_step to 100
            for i in range(4):
                curriculum.compute_reward(f"{base_task_id}_{i}", "<answer>4</answer>")

            # Should have triggered cleanup since global_step % 100 == 0
            # and len(grpo_batch_scores) > 50
            # Cleanup should keep only 20 most recent
            self.assertLessEqual(len(curriculum.grpo_batch_scores), 20)

    def test_grpo_global_step_increments_once_per_batch(self):
        """Test that global_step increments only once per complete batch, not per response."""
        base_task_id = "math_step_test"
        for i in range(4):
            task = Task(
                task_id=f"{base_task_id}_{i}",
                task_name="Step Test",
                task_type="math",
                level=1,
                prompt="What is 2 + 2?",
                expected_answer="4",
            )
            self.curriculum.session.add_task(task)

        with patch.object(
            self.curriculum.reward_functions["math"], "compute_reward"
        ) as mock_compute:
            mock_result = MagicMock()
            mock_result.score = 1.0
            mock_result.info = ""
            mock_compute.return_value = mock_result

            initial_step = self.curriculum.global_step

            # Submit all 4 responses
            for i in range(4):
                self.curriculum.compute_reward(
                    f"{base_task_id}_{i}", "<answer>4</answer>"
                )

            # global_step should have incremented exactly once
            self.assertEqual(self.curriculum.global_step, initial_step + 1)


if __name__ == "__main__":
    unittest.main()
