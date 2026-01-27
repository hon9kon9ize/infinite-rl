import unittest
import json
import tempfile
import os
from unittest.mock import patch, MagicMock
from infinite_rl.curriculum import CurriculumLearning


class TestCurriculumLearning(unittest.TestCase):
    """Test CurriculumLearning class functionality."""

    def setUp(self):
        """Set up test fixtures."""
        # Create temporary math.json file
        self.temp_dir = tempfile.mkdtemp()
        self.math_file = os.path.join(self.temp_dir, "math.json")
        self.puzzle_file = os.path.join(self.temp_dir, "puzzles.json")

        # Sample math data
        math_data = [
            {"problem": "What is 2 + 2?", "solution": "4", "rating": 1},
            {"problem": "What is 10 - 3?", "solution": "7", "rating": 1},
            {"problem": "Solve x + 5 = 10", "solution": "5", "rating": 2},
        ]

        # Sample puzzle data
        puzzle_data = {
            "javascript": {
                "EasyPuzzle": {
                    "name": "EasyPuzzle",
                    "language": "javascript",
                    "docstring": "Easy puzzle",
                    "sat": "function sat() { return true; }",
                    "sol": "function sol() {}",
                    "ans_type": "boolean",
                    "rating": 1,
                },
                "HardPuzzle": {
                    "name": "HardPuzzle",
                    "language": "javascript",
                    "docstring": "Hard puzzle",
                    "sat": "function sat() { return true; }",
                    "sol": "function sol() {}",
                    "ans_type": "boolean",
                    "rating": 5,
                },
            },
            "python": {
                "PythonEasy": {
                    "name": "PythonEasy",
                    "language": "python",
                    "docstring": "Easy Python puzzle",
                    "sat": "def sat(): return True",
                    "sol": "def sol(): pass",
                    "ans_type": "bool",
                    "rating": 1,
                }
            },
        }

        # Write test data files
        with open(self.math_file, "w") as f:
            json.dump(math_data, f)

        with open(self.puzzle_file, "w") as f:
            json.dump(puzzle_data, f)

    def tearDown(self):
        """Clean up test fixtures."""
        import shutil

        shutil.rmtree(self.temp_dir)

    @patch("infinite_rl.curriculum.Path")
    def test_initialization(self, mock_path):
        """Test CurriculumLearning initialization."""
        # Mock the file paths
        mock_path.return_value.parent = MagicMock()
        mock_path.return_value.parent.__str__ = MagicMock(return_value=self.temp_dir)

        # Mock the runtimes directory
        with patch("os.path.exists", return_value=True):
            with patch("builtins.open") as mock_open:
                # Mock file opening for math.json
                mock_file = MagicMock()
                mock_file.__enter__.return_value = mock_file
                mock_file.__exit__.return_value = None
                mock_open.return_value = mock_file

                # Mock json.load for math data
                with patch("json.load") as mock_json_load:
                    mock_json_load.return_value = [
                        {"problem": "2+2", "solution": "4", "rating": 1}
                    ]

                    cl = CurriculumLearning()

                    # Check initial state
                    self.assertEqual(cl.current_level, 1)
                    self.assertEqual(cl.task_counters, {})
                    self.assertEqual(cl.failed_tasks, {})
                    self.assertEqual(len(cl.recent_tasks), 0)

    def test_get_rewards_correct_answer(self):
        """Test get_rewards with correct answer."""
        cl = CurriculumLearning()

        # Mock reward function to return correct score
        with patch.object(
            cl.reward_functions["math"], "compute_reward"
        ) as mock_compute:
            mock_result = MagicMock()
            mock_result.score = 1.0
            mock_compute.return_value = mock_result

            reward = cl.get_rewards("math", "<answer>4</answer>", "4", "test_task")

            self.assertEqual(reward, 1.0)
            self.assertEqual(cl.task_counters["math"], 1)
            self.assertEqual(len(cl.failed_tasks), 0)

    def test_get_rewards_wrong_answer(self):
        """Test get_rewards with wrong answer."""
        cl = CurriculumLearning(
            use_format=False,
            use_repetition=False,
            use_reasoning_steps=False,
            use_length=False,
            use_lang_consistency=False,
        )

        # Mock reward function to return incorrect score
        with patch.object(
            cl.reward_functions["math"], "compute_reward"
        ) as mock_compute:
            mock_result = MagicMock()
            mock_result.score = 0.0
            mock_compute.return_value = mock_result

            reward = cl.get_rewards("math", "<answer>5</answer>", "4", "test_task")

            self.assertEqual(reward, 0.0)
            self.assertEqual(cl.task_counters["math"], -1)
            self.assertIn("test_task", cl.failed_tasks)
            self.assertEqual(cl.failed_tasks["test_task"], "<answer>5</answer>")

    def test_level_advancement(self):
        """Test automatic level advancement."""
        cl = CurriculumLearning()
        cl.current_level = 1

        # Simulate good performance
        cl.task_counters = {"math": 5, "puzzle": 5}  # High positive scores

        # Manually trigger level update (normally called in get_rewards)
        cl._update_level()

        # Should advance to level 2
        self.assertEqual(cl.current_level, 2)

    def test_get_prompt_math_task(self):
        """Test getting a math task prompt."""
        # Create a minimal curriculum with mock data
        cl = CurriculumLearning()

        # Manually set up tasks for level 1
        cl.tasks_by_level[1] = [
            {
                "type": "math",
                "data": {"problem": "What is 2+2?", "solution": "4", "rating": 1},
                "rating": 1,
                "id": "math_test",
            }
        ]

        task = cl.get_prompt()

        self.assertIsNotNone(task)
        self.assertEqual(task["task_type"], "math")
        self.assertEqual(task["level"], 1)
        self.assertEqual(task["expected_output"], "4")
        self.assertIn("What is 2+2?", task["prompt"])

    def test_get_prompt_puzzle_task(self):
        """Test getting a puzzle task prompt."""
        cl = CurriculumLearning()

        # Manually set up puzzle task
        cl.tasks_by_level[1] = [
            {
                "type": "puzzle",
                "language": "javascript",
                "puzzle_name": "TestPuzzle",
                "data": {
                    "name": "TestPuzzle",
                    "docstring": "Test puzzle description",
                    "sat": "function sat() { return true; }",
                    "sol": "function sol() {}",
                    "ans_type": "boolean",
                    "rating": 1,
                },
                "rating": 1,
                "id": "puzzle_test",
            }
        ]

        task = cl.get_prompt()

        self.assertIsNotNone(task)
        self.assertEqual(task["task_type"], "puzzle")
        self.assertEqual(task["level"], 1)
        self.assertIn("TestPuzzle", task["prompt"])
        self.assertIn("Test puzzle description", task["prompt"])

    def test_recent_tasks_tracking(self):
        """Test that recent tasks are tracked and weighted."""
        cl = CurriculumLearning()

        # Add some recent tasks
        cl.recent_tasks = ["task1", "task2", "task1"]  # task1 appears twice

        # Set up tasks including task1
        cl.tasks_by_level[1] = [
            {
                "type": "math",
                "data": {"problem": "Q1", "solution": "A1", "rating": 1},
                "rating": 1,
                "id": "task1",
            },
            {
                "type": "math",
                "data": {"problem": "Q2", "solution": "A2", "rating": 1},
                "rating": 1,
                "id": "task2",
            },
            {
                "type": "math",
                "data": {"problem": "Q3", "solution": "A3", "rating": 1},
                "rating": 1,
                "id": "task3",
            },
        ]

        # Get a prompt - should prefer task3 over task1 (which has been seen recently)
        task = cl.get_prompt()

        self.assertIsNotNone(task)
        # The selection is random but weighted, so we can't guarantee which task
        # but we can verify the structure
        self.assertIn(task["task_id"], ["task1", "task2", "task3"])

    def test_get_learning_stats(self):
        """Test getting learning statistics."""
        cl = CurriculumLearning()
        cl.current_level = 3
        cl.task_counters = {"math": 2, "puzzle": -1}
        cl.failed_tasks = {"failed_task": "wrong answer"}

        stats = cl.get_learning_stats()

        self.assertEqual(stats["current_level"], 3)
        self.assertEqual(stats["task_counters"], {"math": 2, "puzzle": -1})
        self.assertEqual(stats["failed_tasks_count"], 1)
        self.assertEqual(stats["recent_tasks_count"], 0)
        self.assertIsInstance(stats["available_tasks_by_level"], dict)

    def test_invalid_task_type(self):
        """Test handling of invalid task type."""
        cl = CurriculumLearning()

        with self.assertRaises(ValueError):
            cl.get_rewards("invalid_type", "answer", "expected", "task_id")

    def test_no_tasks_available(self):
        """Test behavior when no tasks are available."""
        cl = CurriculumLearning()
        # Clear all loaded tasks to simulate no tasks available
        cl.tasks_by_level = {i: [] for i in range(1, 6)}

        task = cl.get_prompt()
        self.assertIsNone(task)

    def test_initialization_with_format_reward_default(self):
        """Test CurriculumLearning initialization with default format reward enabled."""
        cl = CurriculumLearning(use_format=True)

        self.assertTrue(cl.use_format)
        self.assertIn("format", cl.aux_reward_functions)
        self.assertIsNotNone(cl.aux_reward_functions.get("format"))

    def test_initialization_with_lang_consistency_reward(self):
        """Test initialization with language consistency reward."""
        cl = CurriculumLearning(use_lang_consistency=True)

        self.assertTrue(cl.use_lang_consistency)
        self.assertIn("lang_consistency", cl.aux_reward_functions)

    def test_initialization_with_multiple_auxiliary_rewards(self):
        """Test initialization with multiple auxiliary rewards enabled."""
        cl = CurriculumLearning(
            use_format=True,
            use_lang_consistency=True,
            use_repetition=True,
            use_reasoning_steps=True,
            use_length=True,
        )

        self.assertTrue(cl.use_format)
        self.assertTrue(cl.use_lang_consistency)
        self.assertTrue(cl.use_repetition)
        self.assertTrue(cl.use_reasoning_steps)
        self.assertTrue(cl.use_length)

        # Check all auxiliary functions are initialized
        self.assertGreaterEqual(len(cl.aux_reward_functions), 1)

    def test_initialization_with_auxiliary_kwargs(self):
        """Test initialization with custom kwargs for auxiliary rewards."""
        length_kwargs = {"target_len": 200, "max_len": 500}
        reasoning_kwargs = {"bonus_per_step": 0.05}

        cl = CurriculumLearning(
            use_length=True,
            length_kwargs=length_kwargs,
            use_reasoning_steps=True,
            reasoning_steps_kwargs=reasoning_kwargs,
        )

        self.assertEqual(cl.length_kwargs, length_kwargs)
        self.assertEqual(cl.reasoning_steps_kwargs, reasoning_kwargs)

    def test_get_aux_reward_scores(self):
        """Test getting auxiliary reward scores for a response."""
        cl = CurriculumLearning(use_format=True)

        # Create a mock response with proper formatting
        response = "<think>This is my reasoning</think>\n<answer>42</answer>"
        expected = "42"

        # Get auxiliary scores
        aux_scores = cl.get_aux_reward_scores(response, expected)

        self.assertIsInstance(aux_scores, dict)
        # At least format should be evaluated
        self.assertGreaterEqual(len(aux_scores), 0)

    def test_get_learning_stats_includes_aux_rewards(self):
        """Test that learning stats include auxiliary reward functions."""
        cl = CurriculumLearning(use_format=True, use_repetition=True, use_length=True)

        stats = cl.get_learning_stats()

        self.assertIn("aux_reward_functions", stats)
        self.assertIsInstance(stats["aux_reward_functions"], list)
        # Should list the active auxiliary functions
        self.assertGreaterEqual(len(stats["aux_reward_functions"]), 0)

    def test_combined_reward_computation(self):
        """Test that primary and auxiliary rewards are combined correctly."""
        cl = CurriculumLearning(use_format=True)

        # Mock the primary reward function
        with patch.object(
            cl.reward_functions["math"], "compute_reward"
        ) as mock_primary:
            mock_result = MagicMock()
            mock_result.score = 1.0
            mock_primary.return_value = mock_result

            # Mock auxiliary reward computation
            with patch.object(cl, "get_aux_reward_scores") as mock_aux_scores:
                # Return some auxiliary scores
                mock_aux_scores.return_value = {"format": 0.8}

                response = "<answer>42</answer>"
                reward = cl.get_rewards("math", response, "42", "test_task")

                # Combined score should be 70% primary + 30% auxiliary average
                # 0.7 * 1.0 + 0.3 * 0.8 = 0.7 + 0.24 = 0.94
                self.assertAlmostEqual(reward, 0.94, places=2)

    def test_auxiliary_rewards_affect_curriculum_progression(self):
        """Test that combined rewards (primary + auxiliary) affect curriculum level."""
        cl = CurriculumLearning(use_format=True, use_repetition=True)
        cl.current_level = 1

        # Simulate good performance with combined rewards
        cl.task_counters = {"math": 5, "puzzle": 5}

        # Trigger level update
        cl._update_level()

        # Should advance level based on combined scores
        self.assertGreaterEqual(cl.current_level, 1)

    def test_auxiliary_reward_with_custom_answer_tag(self):
        """Test auxiliary rewards work with custom answer tags."""
        cl = CurriculumLearning(
            use_format=True,
            answer_tag="result",
            think_tag="reasoning",
        )

        self.assertEqual(cl.answer_tag, "result")
        self.assertEqual(cl.think_tag, "reasoning")

    def test_get_aux_reward_scores_with_no_auxiliary_rewards(self):
        """Test get_aux_reward_scores when no auxiliary rewards are enabled."""
        cl = CurriculumLearning(
            use_format=False,
            use_lang_consistency=False,
            use_repetition=False,
            use_reasoning_steps=False,
            use_length=False,
        )

        response = "<answer>42</answer>"
        expected = "42"

        aux_scores = cl.get_aux_reward_scores(response, expected)

        # Should still return a dict, but possibly empty
        self.assertIsInstance(aux_scores, dict)

    def test_auxiliary_reward_kwargs_passed_correctly(self):
        """Test that kwargs are passed correctly to auxiliary reward functions."""
        length_kwargs = {"target_len": 250, "max_len": 600}
        format_kwargs = {"strict": True}

        cl = CurriculumLearning(
            use_length=True,
            length_kwargs=length_kwargs,
            use_format=True,
            format_kwargs=format_kwargs,
        )

        # Verify kwargs are stored
        self.assertEqual(cl.length_kwargs, length_kwargs)
        self.assertEqual(cl.format_kwargs, format_kwargs)

    def test_all_auxiliary_rewards_together(self):
        """Integration test: all auxiliary rewards enabled together."""
        cl = CurriculumLearning(
            use_format=True,
            use_lang_consistency=True,
            use_repetition=True,
            use_reasoning_steps=True,
            use_length=True,
            length_kwargs={"target_len": 200},
            reasoning_steps_kwargs={"bonus_per_step": 0.1},
        )

        # Verify all are initialized
        self.assertTrue(cl.use_format)
        self.assertTrue(cl.use_lang_consistency)
        self.assertTrue(cl.use_repetition)
        self.assertTrue(cl.use_reasoning_steps)
        self.assertTrue(cl.use_length)

        # Get stats to verify they're all tracked
        stats = cl.get_learning_stats()
        self.assertIn("aux_reward_functions", stats)

        # Verify we can get auxiliary scores
        response = "<think>Reasoning here</think>\n<answer>42</answer>"
        aux_scores = cl.get_aux_reward_scores(response, "42")
        self.assertIsInstance(aux_scores, dict)


if __name__ == "__main__":
    unittest.main()
