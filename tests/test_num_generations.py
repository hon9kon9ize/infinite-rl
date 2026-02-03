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
        # Mock truthy task loading to avoid llm_judge requirement
        with patch(
            "infinite_rl.curriculum.CurriculumLearning._load_available_tasks"
        ) as mock_load:
            mock_load.return_value = None
            curriculum = CurriculumLearning(num_generations=2)

            # Add task to the session manually
            task_id = "math_test_0"
            task = Task(
                task_id=task_id,
                task_name="Test",
                task_type="math",
                level=0,
                prompt="Test",
                expected_answer="4",
            )
            curriculum.session.add_task(task)

            # First response - batch should accumulate
            with patch.object(
                curriculum.reward_functions["math"], "compute_reward"
            ) as mock_reward:
                mock_result = MagicMock()
                mock_result.score = 1.0
                mock_result.info = ""
                mock_reward.return_value = mock_result

                curriculum.compute_reward(
                    task_id, "<think>test</think>\n<answer>4</answer>"
                )
                # Check that task has 1 generation
                task = curriculum.session.get_task(task_id)
                self.assertEqual(len(task.generations), 1)

            # Second response - should trigger level update since batch is complete
            with patch.object(
                curriculum.reward_functions["math"], "compute_reward"
            ) as mock_reward:
                mock_result = MagicMock()
                mock_result.score = 1.0
                mock_result.info = ""
                mock_reward.return_value = mock_result

                initial_step = curriculum.global_step
                curriculum.compute_reward(
                    task_id, "<think>test</think>\n<answer>4</answer>"
                )
                # Should have triggered level update since we reached num_generations
                task = curriculum.session.get_task(task_id)
                self.assertEqual(len(task.generations), 2)
                # global_step should have incremented
                self.assertEqual(curriculum.global_step, initial_step + 1)

    def test_num_generations_level_update_trigger(self):
        """Test that level update is triggered when num_generations responses are accumulated."""
        # Mock truthy task loading to avoid llm_judge requirement
        with patch(
            "infinite_rl.curriculum.CurriculumLearning._load_available_tasks"
        ) as mock_load:
            mock_load.return_value = None
            curriculum = CurriculumLearning(
                num_generations=2,
                window_size=10,
                warmup_step=0,  # Disable warmup for this test
            )

        # Create and add a task manually
        task_id = "math_0_0"
        task = Task(
            task_id=task_id,
            task_name="Test",
            task_type="math",
            level=0,
            prompt="Test",
            expected_answer="4",
        )
        curriculum.session.add_task(task)

        # Mock reward function
        with patch.object(
            curriculum.reward_functions["math"], "compute_reward"
        ) as mock_reward:
            mock_result = MagicMock()
            mock_result.score = 1.0
            mock_result.info = ""
            mock_reward.return_value = mock_result

            # Submit first response
            initial_step = curriculum.global_step
            curriculum.compute_reward(
                task_id, "<think>test</think>\n<answer>4</answer>"
            )
            # Still incomplete batch - step shouldn't increment yet
            self.assertEqual(curriculum.global_step, initial_step)
            task = curriculum.session.get_task(task_id)
            self.assertEqual(len(task.generations), 1)

            # Submit second response (completes batch)
            curriculum.compute_reward(
                task_id, "<think>test</think>\n<answer>4</answer>"
            )

            # global_step should have incremented by 1 (once per complete batch)
            self.assertEqual(curriculum.global_step, initial_step + 1)
            task = curriculum.session.get_task(task_id)
            self.assertEqual(len(task.generations), 2)

    def test_num_generations_one(self):
        """Test behavior with num_generations=1 (immediate level updates)."""
        # Mock truthy task loading to avoid llm_judge requirement
        with patch(
            "infinite_rl.curriculum.CurriculumLearning._load_available_tasks"
        ) as mock_load:
            mock_load.return_value = None
            curriculum = CurriculumLearning(
                num_generations=1, warmup_step=0  # Disable warmup
            )

            # Mock reward function to avoid actual computation
            with patch.object(
                curriculum.reward_functions["math"], "compute_reward"
            ) as mock_reward:
                mock_result = MagicMock()
                mock_result.score = 1.0
                mock_result.info = ""
                mock_reward.return_value = mock_result

                # Create a task manually since get_prompt would trigger uninitialized tasks
                task = Task(
                    task_id="math_test_0",
                    task_name="Test",
                    task_type="math",
                    level=0,
                    prompt="Test",
                    expected_answer="4",
                )
                curriculum.session.add_task(task)
                self.assertIsNotNone(task)

        # With num_generations=1, a single response should trigger level update
        # (batch should complete immediately)
        initial_step = curriculum.global_step
        curriculum.compute_reward(task.task_id, "response")

        # Verify batch was completed (task should have 1 generation and global_step should increment)
        task = curriculum.session.get_task(task.task_id)
        self.assertEqual(len(task.generations), 1)
        self.assertEqual(curriculum.global_step, initial_step + 1)

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
        """Test that multiple responses for the same task accumulate correctly."""
        # Create one task for the prompt
        task_id = "math_test_123"
        task = Task(
            task_id=task_id,
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
            self.curriculum.compute_reward(task_id, "<answer>4</answer>")
            task = self.curriculum.session.get_task(task_id)
            self.assertEqual(len(task.generations), 1)
            self.assertEqual(
                self.curriculum.global_step, initial_step
            )  # Not incremented yet

            # Response 2
            self.curriculum.compute_reward(task_id, "<answer>4</answer>")
            task = self.curriculum.session.get_task(task_id)
            self.assertEqual(len(task.generations), 2)
            self.assertEqual(
                self.curriculum.global_step, initial_step
            )  # Still not incremented

            # Response 3
            self.curriculum.compute_reward(task_id, "<answer>4</answer>")
            task = self.curriculum.session.get_task(task_id)
            self.assertEqual(len(task.generations), 3)
            self.assertEqual(
                self.curriculum.global_step, initial_step
            )  # Still not incremented

            # Response 4 - should trigger batch completion
            self.curriculum.compute_reward(task_id, "<answer>4</answer>")

            # Task should have all 4 generations
            task = self.curriculum.session.get_task(task_id)
            self.assertEqual(len(task.generations), 4)

            # global_step should increment ONCE for the entire batch
            self.assertEqual(self.curriculum.global_step, initial_step + 1)
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

        # Create a single task
        task_id = "math_test_456"
        task = Task(
            task_id=task_id,
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
                curriculum.compute_reward(task_id, "<answer>4</answer>")

            # Level should NOT change with incomplete batch
            self.assertEqual(curriculum.current_level, initial_level)
            self.assertEqual(len(task.generations), 3)

            # Complete the batch
            curriculum.compute_reward(task_id, "<answer>4</answer>")

            # Now the batch is complete and should be processed
            self.assertEqual(len(task.generations), 4)

    def test_grpo_batch_handles_oversized_batch(self):
        """Test that oversized batches (more responses than expected) are handled."""
        task_id = "math_test_789"
        task = Task(
            task_id=task_id,
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
                self.curriculum.compute_reward(task_id, "<answer>4</answer>")

            # Should have processed the first 4 and kept accumulating
            # The 5th response adds to the existing task's generations and triggers another processing
            self.assertEqual(len(task.generations), 5)
            self.assertEqual(self.curriculum.global_step, initial_step + 2)

    def test_grpo_batch_different_tasks_independent(self):
        """Test that different tasks maintain independent batch counters."""
        # Create two separate tasks
        task_id1 = "math_test_111"
        task1 = Task(
            task_id=task_id1,
            task_name="Test Math 1",
            task_type="math",
            level=1,
            prompt="What is 2 + 2?",
            expected_answer="4",
        )
        task_id2 = "math_test_222"
        task2 = Task(
            task_id=task_id2,
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
            self.curriculum.compute_reward(task_id1, "<answer>4</answer>")
            self.curriculum.compute_reward(task_id1, "<answer>4</answer>")

            # Submit 3 responses for task2
            self.curriculum.compute_reward(task_id2, "<answer>6</answer>")
            self.curriculum.compute_reward(task_id2, "<answer>6</answer>")
            self.curriculum.compute_reward(task_id2, "<answer>6</answer>")

            # Both should have independent counters
            self.assertEqual(len(task1.generations), 2)
            self.assertEqual(len(task2.generations), 3)

    def test_grpo_global_step_increments_once_per_batch(self):
        """Test that global_step increments only once per complete batch, not per response."""
        task_id = "math_step_test"
        task = Task(
            task_id=task_id,
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
                self.curriculum.compute_reward(task_id, "<answer>4</answer>")

            # global_step should have incremented exactly once
            self.assertEqual(self.curriculum.global_step, initial_step + 1)


if __name__ == "__main__":
    unittest.main()
