"""Unit tests for DynamicCurriculumDataset GRPO batching behavior."""

import unittest
from infinite_rl.dynamic_dataset import DynamicCurriculumDataset
from infinite_rl.curriculum import CurriculumLearning, Task


class TestDynamicCurriculumDataset(unittest.TestCase):
    """Test DynamicCurriculumDataset for proper GRPO batching."""

    def setUp(self):
        """Set up test fixtures."""
        self.curriculum = CurriculumLearning(
            num_generations=4,
            warmup_step=0,
            use_format=False,
            use_repetition=False,
            use_reasoning_steps=False,
            use_length=False,
            use_lang_consistency=False,
        )
        self.dataset = DynamicCurriculumDataset(
            curriculum=self.curriculum, num_samples=100
        )

    def test_dataset_initialization(self):
        """Test that dataset initializes correctly."""
        self.assertEqual(self.dataset.num_samples, 100)
        self.assertEqual(len(self.dataset), 100)
        self.assertEqual(self.dataset.num_generations, 4)
        self.assertEqual(len(self.dataset.task_cache), 0)

    def test_same_task_for_grpo_batch(self):
        """Test that indices 0-3 get the same task (GRPO batch)."""
        # Get first 4 items (one GRPO batch)
        items = [self.dataset[i] for i in range(4)]

        # Extract task_ids from metadata
        task_ids = [item["task_metadata"]["task_id"] for item in items]

        # All 4 responses should use the SAME task
        self.assertEqual(
            len(set(task_ids)),
            1,
            f"Expected 1 unique task, got {len(set(task_ids))}: {task_ids}",
        )

    def test_different_tasks_for_different_batches(self):
        """Test that different GRPO batches get different tasks."""
        # Get items from two different batches
        batch0_items = [self.dataset[i] for i in range(4)]  # batch_idx=0
        batch1_items = [self.dataset[i] for i in range(4, 8)]  # batch_idx=1

        batch0_task_id = batch0_items[0]["task_metadata"]["task_id"]
        batch1_task_id = batch1_items[0]["task_metadata"]["task_id"]

        # Different batches should have different tasks
        self.assertNotEqual(
            batch0_task_id,
            batch1_task_id,
            "Different GRPO batches should have different task IDs",
        )

    def test_task_cache_usage(self):
        """Test that task cache properly stores and reuses tasks."""
        # Access first item (creates cache entry for batch_idx=0)
        _ = self.dataset[0]
        self.assertEqual(len(self.dataset.task_cache), 1)
        self.assertIn(0, self.dataset.task_cache)

        # Access another item in same batch (reuses cache)
        _ = self.dataset[1]
        self.assertEqual(len(self.dataset.task_cache), 1)

        # Access item from different batch (creates new cache entry)
        _ = self.dataset[4]
        self.assertEqual(len(self.dataset.task_cache), 2)
        self.assertIn(0, self.dataset.task_cache)
        self.assertIn(1, self.dataset.task_cache)

    def test_task_cache_cleanup(self):
        """Test that task cache is cleaned up to prevent memory leak."""
        # Access enough batches to trigger cleanup (> 20 batches)
        for batch_idx in range(25):
            _ = self.dataset[batch_idx * 4]  # First item of each batch

        # Cache should be limited to 20 entries
        self.assertLessEqual(
            len(self.dataset.task_cache),
            20,
            f"Cache should be <= 20, got {len(self.dataset.task_cache)}",
        )

        # Most recent batch should still be in cache
        self.assertIn(24, self.dataset.task_cache)

    def test_batch_index_calculation(self):
        """Test that batch_idx is calculated correctly for different indices."""
        test_cases = [
            (0, 0),  # idx=0 -> batch_idx=0
            (1, 0),  # idx=1 -> batch_idx=0
            (3, 0),  # idx=3 -> batch_idx=0
            (4, 1),  # idx=4 -> batch_idx=1
            (7, 1),  # idx=7 -> batch_idx=1
            (8, 2),  # idx=8 -> batch_idx=2
            (15, 3),  # idx=15 -> batch_idx=3
        ]

        for idx, expected_batch_idx in test_cases:
            calculated_batch_idx = idx // self.dataset.num_generations
            self.assertEqual(
                calculated_batch_idx,
                expected_batch_idx,
                f"idx={idx} should map to batch_idx={expected_batch_idx}, got {calculated_batch_idx}",
            )

    def test_dataset_output_format(self):
        """Test that dataset returns correctly formatted data."""
        item = self.dataset[0]

        # Check structure
        self.assertIn("prompt", item)
        self.assertIn("task_metadata", item)

        # Check prompt format (should be list of message dicts)
        self.assertIsInstance(item["prompt"], list)
        self.assertEqual(len(item["prompt"]), 1)
        self.assertEqual(item["prompt"][0]["role"], "user")
        self.assertIn("content", item["prompt"][0])

        # Check task_metadata
        metadata = item["task_metadata"]
        required_fields = [
            "task_id",
            "task_name",
            "task_type",
            "level",
            "language",
            "expected_answer",
        ]
        for field in required_fields:
            self.assertIn(field, metadata, f"Missing required field: {field}")

    def test_all_responses_same_prompt_content(self):
        """Test that all responses in a GRPO batch have identical prompt content."""
        batch_items = [self.dataset[i] for i in range(4)]

        prompt_contents = [item["prompt"][0]["content"] for item in batch_items]

        # All prompts in the batch should be identical
        self.assertEqual(
            len(set(prompt_contents)),
            1,
            "All responses in a GRPO batch should have the same prompt content",
        )

    def test_num_generations_two(self):
        """Test dataset behavior with num_generations=2."""
        curriculum = CurriculumLearning(
            num_generations=2,
            warmup_step=0,
            use_format=False,
            use_repetition=False,
            use_reasoning_steps=False,
            use_length=False,
            use_lang_consistency=False,
        )
        dataset = DynamicCurriculumDataset(curriculum=curriculum, num_samples=100)

        # Indices 0-1 should get the same task
        task0 = dataset[0]["task_metadata"]["task_id"]
        task1 = dataset[1]["task_metadata"]["task_id"]
        self.assertEqual(task0, task1)

        # Index 2 should get a different task (new batch)
        task2 = dataset[2]["task_metadata"]["task_id"]
        self.assertNotEqual(task0, task2)

    def test_integration_with_curriculum(self):
        """Test that dataset properly integrates with curriculum task generation."""
        # Get multiple items
        items = [self.dataset[i] for i in range(8)]  # 2 batches

        # All should have valid task IDs
        for item in items:
            task_id = item["task_metadata"]["task_id"]
            self.assertIsNotNone(task_id)
            self.assertGreater(len(task_id), 0)

        # Should have created 2 unique tasks (2 batches * 1 task per batch)
        unique_task_ids = set(item["task_metadata"]["task_id"] for item in items)
        self.assertEqual(
            len(unique_task_ids),
            2,
            f"Expected 2 unique tasks, got {len(unique_task_ids)}: {unique_task_ids}",
        )


if __name__ == "__main__":
    unittest.main()
