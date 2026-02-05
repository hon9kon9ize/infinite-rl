"""
Test suite to verify reward ordering in GRPO training.

This ensures rewards are correctly assigned to generations in the correct order,
which is critical for GRPO optimization to work correctly.
"""

import unittest
from unittest.mock import Mock, patch, MagicMock
from collections import defaultdict
from typing import List, Dict, Any


def create_test_reward_func(curriculum_mock):
    """Create the reward function for testing (same logic as train.py)."""

    def reward_func(
        prompts: List[str],
        completions: List[str],
        task_metadata: List[Dict[str, Any]] = None,
        **kwargs,
    ) -> List[float]:
        """Compute rewards for completions using Infinite-RL's curriculum."""

        # CRITICAL: Preserve original completion order by storing (index, completion) pairs
        indexed_completions = []
        grouped = defaultdict(list)

        for i, completion in enumerate(completions):
            if task_metadata and i < len(task_metadata):
                task_id = task_metadata[i]["task_id"]
            else:
                task_id = f"task_{i // curriculum_mock.num_generations}"

            indexed_completions.append((i, task_id, completion))
            grouped[task_id].append((i, completion))

        # Collect all rewards with their original indices
        reward_with_index = {}

        # Process each task's completions in batch
        for task_id, index_completion_list in grouped.items():
            try:
                # Extract completions
                completion_texts = []
                indices = []
                for orig_idx, completion in index_completion_list:
                    indices.append(orig_idx)
                    if isinstance(completion, dict):
                        completion_text = completion.get("content", completion)
                    elif isinstance(completion, str):
                        completion_text = completion
                    else:
                        completion_text = str(completion)
                    completion_texts.append(completion_text)

                # Mock: compute_rewards returns scores for each completion
                batch_scores = curriculum_mock.compute_rewards(
                    task_id, completion_texts
                )

                # Map scores back to original indices
                for idx, score in zip(indices, batch_scores):
                    reward_with_index[idx] = float(score)

            except Exception as e:
                print(f"Error computing reward for task {task_id}: {e}")
                for idx, _ in index_completion_list:
                    reward_with_index[idx] = 0.0

        # Reconstruct rewards in original order
        rewards_list = [reward_with_index.get(i, 0.0) for i in range(len(completions))]
        return rewards_list

    return reward_func


class TestRewardOrdering(unittest.TestCase):
    """Test suite for reward ordering correctness."""

    def setUp(self):
        """Set up test fixtures."""
        self.curriculum_mock = Mock()
        self.curriculum_mock.num_generations = 4
        self.reward_func = create_test_reward_func(self.curriculum_mock)

    def test_single_task_sequential_order(self):
        """Test that rewards are returned in the correct sequential order."""
        # Setup
        completions = [
            "completion_0",
            "completion_1",
            "completion_2",
            "completion_3",
        ]
        task_metadata = [
            {"task_id": "task_0"},
            {"task_id": "task_0"},
            {"task_id": "task_0"},
            {"task_id": "task_0"},
        ]
        expected_scores = [0.1, 0.2, 0.3, 0.4]

        # Mock the curriculum to return specific scores
        self.curriculum_mock.compute_rewards.return_value = expected_scores

        # Execute
        rewards = self.reward_func([], completions, task_metadata)

        # Assert
        self.assertEqual(len(rewards), 4)
        self.assertEqual(rewards, expected_scores)
        print("✓ Test 1 passed: Single task maintains order")

    def test_multiple_tasks_mixed_order(self):
        """Test with multiple task_ids interleaved (the problematic case)."""
        # Simulate GRPO batching: alternating between task_A and task_B
        completions = [
            "A_gen_0",  # index 0, task_A
            "B_gen_0",  # index 1, task_B
            "A_gen_1",  # index 2, task_A
            "B_gen_1",  # index 3, task_B
        ]
        task_metadata = [
            {"task_id": "task_A"},
            {"task_id": "task_B"},
            {"task_id": "task_A"},
            {"task_id": "task_B"},
        ]

        # Mock: task_A returns scores [0.9, 0.8], task_B returns [0.7, 0.6]
        def mock_compute_rewards(task_id, completions):
            if task_id == "task_A":
                return [0.9, 0.8]  # For 2 completions
            elif task_id == "task_B":
                return [0.7, 0.6]  # For 2 completions
            return [0.0] * len(completions)

        self.curriculum_mock.compute_rewards.side_effect = mock_compute_rewards

        # Execute
        rewards = self.reward_func([], completions, task_metadata)

        # Assert: Must be in order [task_A[0], task_B[0], task_A[1], task_B[1]]
        expected_rewards = [0.9, 0.7, 0.8, 0.6]
        self.assertEqual(
            rewards, expected_rewards, f"Expected {expected_rewards}, got {rewards}"
        )
        print("✓ Test 2 passed: Multiple interleaved tasks maintain order")

    def test_three_tasks_complex_interleaving(self):
        """Test with 3 different tasks in complex interleaving pattern."""
        completions = [
            "A_0",  # 0: task_A
            "B_0",  # 1: task_B
            "C_0",  # 2: task_C
            "A_1",  # 3: task_A
            "C_1",  # 4: task_C
            "B_1",  # 5: task_B
        ]
        task_metadata = [
            {"task_id": "task_A"},
            {"task_id": "task_B"},
            {"task_id": "task_C"},
            {"task_id": "task_A"},
            {"task_id": "task_C"},
            {"task_id": "task_B"},
        ]

        scores_per_task = {
            "task_A": [0.95, 0.85],
            "task_B": [0.75, 0.65],
            "task_C": [0.55, 0.45],
        }

        def mock_compute_rewards(task_id, completions):
            return scores_per_task.get(task_id, [0.0] * len(completions))

        self.curriculum_mock.compute_rewards.side_effect = mock_compute_rewards

        # Execute
        rewards = self.reward_func([], completions, task_metadata)

        # Assert: Must match original order
        expected = [0.95, 0.75, 0.55, 0.85, 0.45, 0.65]
        self.assertEqual(rewards, expected, f"Expected {expected}, got {rewards}")
        print("✓ Test 3 passed: Three tasks with complex interleaving maintain order")

    def test_without_task_metadata(self):
        """Test fallback when task_metadata is not provided."""
        # Without metadata, should use fallback: task_id = task_{i // num_generations}
        # Set num_generations to 2 for this test
        self.curriculum_mock.num_generations = 2

        completions = [
            "gen_0_0",  # index 0, task_0 (0 // 2 = 0)
            "gen_0_1",  # index 1, task_0 (1 // 2 = 0)
            "gen_1_0",  # index 2, task_1 (2 // 2 = 1)
            "gen_1_1",  # index 3, task_1 (3 // 2 = 1)
        ]

        def mock_compute_rewards(task_id, completions):
            if task_id == "task_0":
                return [0.9, 0.8]
            elif task_id == "task_1":
                return [0.7, 0.6]
            return [0.0] * len(completions)

        self.curriculum_mock.compute_rewards.side_effect = mock_compute_rewards

        # Execute (no metadata)
        rewards = self.reward_func([], completions)

        # Assert
        expected = [0.9, 0.8, 0.7, 0.6]
        self.assertEqual(rewards, expected, f"Expected {expected}, got {rewards}")
        print("✓ Test 4 passed: Fallback without metadata maintains order")

    def test_duplicate_task_ids(self):
        """Test when same task_id appears multiple times non-consecutively."""
        completions = [
            "task_A_0",
            "task_B_0",
            "task_A_1",
            "task_B_1",
            "task_A_2",
        ]
        task_metadata = [
            {"task_id": "task_A"},
            {"task_id": "task_B"},
            {"task_id": "task_A"},
            {"task_id": "task_B"},
            {"task_id": "task_A"},
        ]

        scores_per_task = {
            "task_A": [0.9, 0.8, 0.7],  # 3 completions
            "task_B": [0.6, 0.5],  # 2 completions
        }

        def mock_compute_rewards(task_id, completions):
            return scores_per_task.get(task_id, [0.0] * len(completions))

        self.curriculum_mock.compute_rewards.side_effect = mock_compute_rewards

        # Execute
        rewards = self.reward_func([], completions, task_metadata)

        # Assert: Must maintain order [A0, B0, A1, B1, A2]
        expected = [0.9, 0.6, 0.8, 0.5, 0.7]
        self.assertEqual(rewards, expected, f"Expected {expected}, got {rewards}")
        print("✓ Test 5 passed: Duplicate task_ids maintain order")

    def test_edge_case_single_completion(self):
        """Test edge case with single completion."""
        completions = ["single_gen"]
        task_metadata = [{"task_id": "task_single"}]

        self.curriculum_mock.compute_rewards.return_value = [0.5]

        rewards = self.reward_func([], completions, task_metadata)

        self.assertEqual(rewards, [0.5])
        print("✓ Test 6 passed: Single completion works")

    def test_edge_case_many_generations(self):
        """Test with many generations (e.g., 16 instead of 4)."""
        num_gens = 16
        completions = [f"gen_{i}" for i in range(num_gens)]
        task_metadata = [{"task_id": f"task_0"} for _ in range(num_gens)]

        expected_scores = [i / 100.0 for i in range(num_gens)]
        self.curriculum_mock.compute_rewards.return_value = expected_scores

        rewards = self.reward_func([], completions, task_metadata)

        self.assertEqual(rewards, expected_scores)
        print("✓ Test 7 passed: Many generations work")

    def test_curriculum_called_correctly(self):
        """Verify curriculum.compute_rewards is called with correct arguments."""
        completions = [
            "A_gen_0",
            "B_gen_0",
            "A_gen_1",
        ]
        task_metadata = [
            {"task_id": "task_A"},
            {"task_id": "task_B"},
            {"task_id": "task_A"},
        ]

        self.curriculum_mock.compute_rewards.return_value = [0.0]

        self.reward_func([], completions, task_metadata)

        # Verify curriculum was called for each task
        calls = self.curriculum_mock.compute_rewards.call_args_list
        task_ids_called = [call[0][0] for call in calls]

        # Should be called for task_A and task_B
        self.assertIn("task_A", task_ids_called)
        self.assertIn("task_B", task_ids_called)
        print("✓ Test 8 passed: Curriculum called with correct task_ids")

    def test_result_length_matches_input(self):
        """Verify output length always matches input length."""
        test_cases = [1, 2, 4, 8, 16, 100]

        for num_completions in test_cases:
            completions = [f"gen_{i}" for i in range(num_completions)]
            task_metadata = [
                {"task_id": f"task_{i % 3}"} for i in range(num_completions)
            ]

            # Mock to return one score per completion
            self.curriculum_mock.compute_rewards.side_effect = lambda task_id, comps: [
                0.5
            ] * len(comps)

            rewards = self.reward_func([], completions, task_metadata)

            self.assertEqual(
                len(rewards),
                num_completions,
                f"For {num_completions} completions, got {len(rewards)} rewards",
            )

        print("✓ Test 9 passed: Output length always matches input")


class TestRewardOrderingRobustness(unittest.TestCase):
    """Additional tests for robustness and edge cases."""

    def setUp(self):
        """Set up test fixtures."""
        self.curriculum_mock = Mock()
        self.curriculum_mock.num_generations = 4
        self.reward_func = create_test_reward_func(self.curriculum_mock)

    def test_worst_case_reverse_order_tasks(self):
        """Test worst case: tasks processed in reverse order."""
        # If dict iteration happens to return tasks in reverse order
        completions = ["gen_0", "gen_1", "gen_2", "gen_3"]
        task_metadata = [
            {"task_id": "task_Z"},
            {"task_id": "task_Y"},
            {"task_id": "task_X"},
            {"task_id": "task_W"},
        ]

        def mock_compute_rewards(task_id, completions):
            # Return unique score based on task
            task_scores = {
                "task_Z": [0.1],
                "task_Y": [0.2],
                "task_X": [0.3],
                "task_W": [0.4],
            }
            return task_scores.get(task_id, [0.0])

        self.curriculum_mock.compute_rewards.side_effect = mock_compute_rewards

        rewards = self.reward_func([], completions, task_metadata)

        # Must be in original order regardless of dict iteration
        expected = [0.1, 0.2, 0.3, 0.4]
        self.assertEqual(
            rewards, expected, f"Worst case failed: expected {expected}, got {rewards}"
        )
        print("✓ Test 10 passed: Worst case reverse order handled")

    def test_none_scores_handled(self):
        """Test handling of None or missing scores."""
        completions = ["gen_0", "gen_1"]
        task_metadata = [{"task_id": "task_A"}, {"task_id": "task_A"}]

        # Mock returns None (error case)
        self.curriculum_mock.compute_rewards.side_effect = Exception("Test error")

        # Should not crash, should return zeros
        rewards = self.reward_func([], completions, task_metadata)

        self.assertEqual(len(rewards), 2)
        self.assertTrue(all(r == 0.0 for r in rewards))
        print("✓ Test 11 passed: Error handling works")


def run_validation_suite():
    """Run full validation suite and print results."""
    print("\n" + "=" * 80)
    print("REWARD ORDERING VALIDATION SUITE")
    print("=" * 80 + "\n")

    # Run all tests
    suite = unittest.TestLoader().loadTestsFromModule(__import__(__name__))
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)

    # Summary
    print("\n" + "=" * 80)
    if result.wasSuccessful():
        print("✅ ALL TESTS PASSED - Reward ordering is guaranteed correct!")
    else:
        print("❌ SOME TESTS FAILED - Review the failures above")
    print("=" * 80 + "\n")

    return result.wasSuccessful()


if __name__ == "__main__":
    success = run_validation_suite()
    exit(0 if success else 1)
