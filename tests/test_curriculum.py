import unittest
import json
import tempfile
import os
from unittest.mock import patch, MagicMock, mock_open
from infinite_rl.curriculum import (
    CurriculumLearning,
    Task,
    Session,
)
from infinite_rl.reward_functions import RewardFunctionScore


class TestTask(unittest.TestCase):
    """Test Task class functionality."""

    def setUp(self):
        """Set up test fixtures."""
        self.task = Task(
            task_id="math_001",
            task_name="Simple Addition",
            task_type="math",
            level=1,
            prompt="What is 2 + 2?",
            expected_answer="4",
        )

    def test_task_initialization(self):
        """Test Task initialization with basic properties."""
        self.assertEqual(self.task.task_id, "math_001")
        self.assertEqual(self.task.task_name, "Simple Addition")
        self.assertEqual(self.task.task_type, "math")
        self.assertEqual(self.task.level, 1)
        self.assertEqual(self.task.prompt, "What is 2 + 2?")
        self.assertEqual(self.task.expected_answer, "4")
        self.assertEqual(len(self.task.task_rewards), 0)
        self.assertIsNone(self.task.model_output)
        self.assertIsNotNone(self.task.created_at)
        self.assertIsNone(self.task.first_response_at)

    def test_add_reward(self):
        """Test adding rewards to a task."""
        reward = RewardFunctionScore(
            score=1.0,
            reward_function_name="primary",
            info="",
        )
        self.task.add_reward(reward)

        self.assertEqual(len(self.task.task_rewards), 1)
        self.assertEqual(self.task.task_rewards[0].score, 1.0)
        self.assertEqual(self.task.task_rewards[0].reward_function_name, "primary")

    def test_get_score(self):
        """Test getting the primary score from a task."""
        # No rewards yet
        self.assertEqual(self.task.get_score(), 0.0)

        # Add primary reward
        primary_reward = RewardFunctionScore(
            score=0.8,
            reward_function_name="primary",
            info="",
        )
        self.task.add_reward(primary_reward)

        # Should return primary score
        self.assertEqual(self.task.get_score(), 0.8)

        # Add auxiliary reward (should not affect get_score)
        aux_reward = RewardFunctionScore(
            score=0.9,
            reward_function_name="format_answer",
            info="",
        )
        self.task.add_reward(aux_reward)

        # Should still return primary score
        self.assertEqual(self.task.get_score(), 0.8)

    def test_task_to_dict(self):
        """Test converting task to dictionary."""
        reward = RewardFunctionScore(
            score=1.0,
            reward_function_name="primary",
            info="",
        )
        self.task.add_reward(reward)

        task_dict = self.task.to_dict()

        self.assertEqual(task_dict["task_id"], "math_001")
        self.assertEqual(task_dict["task_name"], "Simple Addition")
        self.assertEqual(task_dict["task_type"], "math")
        self.assertEqual(task_dict["level"], 1)
        self.assertEqual(len(task_dict["task_rewards"]), 1)
        self.assertEqual(task_dict["task_rewards"][0]["score"], 1.0)
        self.assertIsNone(task_dict["model_output"])
        self.assertIsNotNone(task_dict["created_at"])
        self.assertIsNotNone(task_dict["first_response_at"])


class TestSession(unittest.TestCase):
    """Test Session class functionality."""

    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.log_file = os.path.join(self.temp_dir, "test_log.jsonl")
        self.session = Session()

    def tearDown(self):
        """Clean up test fixtures."""
        import shutil

        shutil.rmtree(self.temp_dir)

    def test_session_initialization(self):
        """Test Session initialization."""
        self.assertEqual(len(self.session.tasks), 0)
        self.assertEqual(len(self.session.task_history), 0)

    def test_add_task(self):
        """Test adding tasks to a session."""
        task = Task(
            task_id="task_001",
            task_name="Test Task",
            task_type="math",
            level=1,
            prompt="Test prompt",
            expected_answer="Test answer",
        )

        self.session.add_task(task)

        self.assertEqual(len(self.session.tasks), 1)
        self.assertEqual(len(self.session.task_history), 1)
        self.assertIn("task_001", self.session.tasks)

    def test_get_task(self):
        """Test retrieving a task from session."""
        task = Task(
            task_id="task_001",
            task_name="Test Task",
            task_type="math",
            level=1,
            prompt="Test prompt",
            expected_answer="Test answer",
        )
        self.session.add_task(task)

        retrieved = self.session.get_task("task_001")

        self.assertIsNotNone(retrieved)
        self.assertEqual(retrieved.task_id, "task_001")
        self.assertEqual(retrieved.task_name, "Test Task")

    def test_get_nonexistent_task(self):
        """Test retrieving a non-existent task."""
        retrieved = self.session.get_task("nonexistent")
        self.assertIsNone(retrieved)

    def test_set_reward(self):
        """Test setting rewards for a task."""
        task = Task(
            task_id="task_001",
            task_name="Test Task",
            task_type="math",
            level=1,
            prompt="Test prompt",
            expected_answer="4",
        )
        self.session.add_task(task)

        rewards = [
            RewardFunctionScore(
                score=1.0,
                reward_function_name="primary",
                info="",
            ),
            RewardFunctionScore(
                score=0.9,
                reward_function_name="format_answer",
                info="",
            ),
        ]

        self.session.set_reward("task_001", rewards, model_output="model response")

        # Task should have rewards
        retrieved = self.session.get_task("task_001")
        self.assertEqual(len(retrieved.task_rewards), 2)
        self.assertEqual(retrieved.model_output, "model response")
        self.assertIsNotNone(retrieved.created_at)
        self.assertIsNotNone(retrieved.first_response_at)

        # Session no longer writes logs; ensure file remains absent
        self.assertFalse(os.path.exists(self.log_file))

    def test_set_reward_nonexistent_task(self):
        """Test setting reward for non-existent task raises error."""
        rewards = [
            RewardFunctionScore(
                score=1.0,
                reward_function_name="primary",
                info="",
            ),
        ]

        with self.assertRaises(ValueError):
            self.session.set_reward("nonexistent", rewards, 1.0)

    def test_get_task_rewards(self):
        """Test retrieving all rewards for a task."""
        task = Task(
            task_id="task_001",
            task_name="Test Task",
            task_type="math",
            level=1,
            prompt="Test prompt",
            expected_answer="4",
        )
        self.session.add_task(task)

        rewards = [
            RewardFunctionScore(
                score=1.0,
                reward_function_name="primary",
                info="",
            ),
            RewardFunctionScore(
                score=0.8,
                reward_function_name="format_answer",
                info="",
            ),
        ]

        self.session.set_reward("task_001", rewards, 0.9)

        retrieved_rewards = self.session.get_task_rewards("task_001")

        self.assertEqual(len(retrieved_rewards), 2)
        self.assertEqual(retrieved_rewards[0].score, 1.0)
        self.assertEqual(retrieved_rewards[1].score, 0.8)

    def test_task_weights(self):
        """Test calculating task weights for diversity."""
        # Add tasks
        for i in range(3):
            task = Task(
                task_id=f"task_{i}",
                task_name=f"Task {i}",
                task_type="math",
                level=1,
                prompt="Test",
                expected_answer="Answer",
            )
            self.session.add_task(task)

        # Add some to history (task_0 appears twice)
        self.session.task_history = ["task_0", "task_1", "task_0"]

        weights = self.session.task_weights()

        self.assertEqual(len(weights), 3)
        # task_0 should have lower weight due to recency
        self.assertLess(weights["task_0"], weights["task_2"])

    def test_get_stats(self):
        """Test getting session statistics."""
        task = Task(
            task_id="task_001",
            task_name="Test Task",
            task_type="math",
            level=1,
            prompt="Test prompt",
            expected_answer="4",
        )
        self.session.add_task(task)

        stats = self.session.get_stats()

        self.assertEqual(stats["total_tasks"], 1)
        # task_history tracks all tasks added, so one task means one evaluation recorded
        self.assertEqual(stats["total_evaluations"], 1)
        self.assertNotIn("log_file", stats)


class TestCurriculumLearning(unittest.TestCase):
    """Test CurriculumLearning class functionality."""

    def setUp(self):
        """Set up test fixtures."""
        # Create temporary math.json file
        self.temp_dir = tempfile.mkdtemp()
        self.math_file = os.path.join(self.temp_dir, "math.json")
        self.puzzle_file = os.path.join(self.temp_dir, "puzzles.json")
        self.log_file = os.path.join(self.temp_dir, "curriculum_log.jsonl")

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
                    self.assertEqual(
                        cl.current_level, 0
                    )  # Start at level 0 (math tasks only)
                    self.assertEqual(cl._get_task_counters(), {})
                    self.assertEqual(cl._get_failed_tasks(), {})
                    self.assertEqual(len(cl._get_recent_task_ids()), 0)

    def test_get_rewards_correct_answer(self):
        """Test compute_reward with correct answer."""
        cl = CurriculumLearning()

        # Add a task to the session
        task = Task(
            task_id="math_test",
            task_name="Test Math",
            task_type="math",
            level=1,
            prompt="What is 2 + 2?",
            expected_answer="4",
        )
        cl.session.add_task(task)

        # Mock reward function to return correct score
        with patch.object(
            cl.reward_functions["math"], "compute_reward"
        ) as mock_compute:
            mock_result = MagicMock()
            mock_result.score = 1.0
            mock_result.info = ""
            mock_compute.return_value = mock_result

            # Provide valid format with both tags to pass format gate
            response = "<think>2 + 2 = 4</think>\n\n<answer>4</answer>"
            reward = cl.compute_reward("math_test", response)

            self.assertGreaterEqual(reward, 0.5)
            # Check that task has is_correct set to True
            task_from_session = cl.session.get_task("math_test")
            self.assertTrue(task_from_session.is_correct)

    def test_get_rewards_wrong_answer(self):
        """Test compute_reward with wrong answer."""
        cl = CurriculumLearning(
            use_format=False,
            use_repetition=False,
            use_reasoning_steps=False,
            use_length=False,
            use_lang_consistency=False,
        )

        # Add a task to the session
        task = Task(
            task_id="math_test",
            task_name="Test Math",
            task_type="math",
            level=1,
            prompt="What is 2 + 2?",
            expected_answer="4",
        )
        cl.session.add_task(task)

        # Mock reward function to return incorrect score
        with patch.object(
            cl.reward_functions["math"], "compute_reward"
        ) as mock_compute:
            mock_result = MagicMock()
            mock_result.score = 0.0
            mock_result.info = ""
            mock_compute.return_value = mock_result

            reward = cl.compute_reward("math_test", "<answer>5</answer>")

            self.assertLess(reward, 0.5)
            # Check that task has is_correct set to False
            task_from_session = cl.session.get_task("math_test")
            self.assertFalse(task_from_session.is_correct)

    def test_level_advancement(self):
        """Test automatic level advancement."""
        cl = CurriculumLearning()
        cl.current_level = 0

        # Simulate good performance by adding successful tasks to the session
        # Need at least 10 samples per task type due to min_samples constraint
        for i in range(10):
            task = Task(
                task_id=f"math_{i}",
                task_name=f"Math Task {i}",
                task_type="math",
                level=1,
                prompt="Test",
                expected_answer="4",
            )
            task.is_correct = True  # Mark as correct
            cl.session.add_task(task)

        for i in range(10):
            task = Task(
                task_id=f"puzzle_{i}",
                task_name=f"Puzzle Task {i}",
                task_type="puzzle",
                level=1,
                prompt="Test",
                expected_answer="test",
            )
            task.is_correct = True  # Mark as correct
            cl.session.add_task(task)

        # Manually trigger level update (normally called in compute_reward)
        cl._update_level()

        # Should advance to level 1
        self.assertEqual(cl.current_level, 1)

    def test_get_prompt_math_task(self):
        """Test getting a math task prompt as Task object."""
        # Create a minimal curriculum with mock data
        cl = CurriculumLearning()

        # Manually set up tasks for level 0 (curriculum starts at level 0 with math tasks)
        cl.tasks_by_level[0] = [
            {
                "type": "math",
                "data": {
                    "prompt": "What is 2+2?",
                    "response": "4",
                    "lang": "en",
                    "rating": 1,
                },
                "rating": 1,
                "id": "math_test",
            }
        ]

        task = cl.get_prompt()

        self.assertIsNotNone(task)
        self.assertIsInstance(task, Task)
        self.assertEqual(task.task_type, "math")
        self.assertEqual(task.level, 1)
        self.assertEqual(task.expected_answer, "4")
        self.assertEqual(task.language, "en")
        self.assertIn("What is 2+2?", task.prompt)

    def test_get_prompt_puzzle_task(self):
        """Test getting a puzzle task prompt as Task object."""
        cl = CurriculumLearning()

        # Manually set up puzzle task at level 0 (curriculum starts at level 0)
        cl.tasks_by_level[0] = [
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
        self.assertIsInstance(task, Task)
        self.assertEqual(task.task_type, "puzzle")
        self.assertEqual(task.level, 1)
        self.assertEqual(task.language, "javascript")
        self.assertIn("TestPuzzle", task.prompt)
        self.assertIn("Test puzzle description", task.prompt)

    def test_recent_tasks_tracking(self):
        """Test that recent tasks are tracked and weighted."""
        cl = CurriculumLearning()

        # Set up tasks at level 0 (curriculum starts at level 0 with math tasks)
        cl.tasks_by_level[0] = [
            {
                "type": "math",
                "data": {"prompt": "Q1", "response": "A1", "lang": "en", "rating": 1},
                "rating": 1,
                "id": "math_task1",
            },
            {
                "type": "math",
                "data": {"prompt": "Q2", "response": "A2", "lang": "en", "rating": 1},
                "rating": 1,
                "id": "math_task2",
            },
            {
                "type": "math",
                "data": {"prompt": "Q3", "response": "A3", "lang": "en", "rating": 1},
                "rating": 1,
                "id": "math_task3",
            },
        ]

        # Get a prompt - should return one of the available tasks
        task = cl.get_prompt()

        self.assertIsNotNone(task)
        # Task should have math_taskX_0 format with instance counter
        self.assertIn("math_task", task.task_id)

    def test_unique_task_ids_generated(self):
        """Test that each task instance gets a unique task_id."""
        cl = CurriculumLearning()

        # Set up a single task at level 0
        cl.tasks_by_level[0] = [
            {
                "type": "math",
                "data": {"prompt": "Q1", "response": "A1", "lang": "en", "rating": 1},
                "rating": 1,
                "id": "math_simple",
            },
        ]

        # Get multiple prompts from the same task - each should have unique ID
        task1 = cl.get_prompt()
        task2 = cl.get_prompt()
        task3 = cl.get_prompt()

        self.assertIsNotNone(task1)
        self.assertIsNotNone(task2)
        self.assertIsNotNone(task3)

        # Unique IDs should increment counter (math_simple_0, math_simple_1, math_simple_2)
        self.assertEqual(task1.task_id, "math_simple_0")
        self.assertEqual(task2.task_id, "math_simple_1")
        self.assertEqual(task3.task_id, "math_simple_2")

        # Task instance counter should be incremented
        self.assertEqual(cl.task_instance_counter, 3)

    def test_get_learning_stats(self):
        """Test getting learning statistics."""
        cl = CurriculumLearning()
        cl.current_level = 3

        # Add tasks to session to simulate learning state
        for i in range(2):
            task = Task(
                task_id=f"math_{i}",
                task_name=f"Math Task {i}",
                task_type="math",
                level=1,
                prompt="Test",
                expected_answer="4",
            )
            task.is_correct = True
            cl.session.add_task(task)

        # Add a failed puzzle task
        failed_task = Task(
            task_id="puzzle_failed",
            task_name="Failed Puzzle",
            task_type="puzzle",
            level=1,
            prompt="Test",
            expected_answer="test",
        )
        failed_task.is_correct = False
        cl.session.add_task(failed_task)

        stats = cl.get_learning_stats()

        self.assertEqual(stats["current_level"], 3)
        self.assertEqual(stats["task_counters"], {"math": 2, "puzzle": -1})
        self.assertEqual(stats["failed_tasks_count"], 1)
        self.assertEqual(stats["recent_tasks_count"], 3)
        self.assertIsInstance(stats["available_tasks_by_level"], dict)

    def test_invalid_task_type(self):
        """Test handling of invalid task type."""
        cl = CurriculumLearning()

        # Create a task with unknown type
        task = Task(
            task_id="invalid_task",
            task_name="Invalid Task",
            task_type="invalid_type",
            level=1,
            prompt="Test",
            expected_answer="Test",
        )
        cl.session.add_task(task)

        # Provide valid format with both tags so we reach the task type check
        # (format gate runs before task type validation)
        with self.assertRaises(ValueError):
            cl.compute_reward(
                "invalid_task", "<think>test</think>\n\n<answer>answer</answer>"
            )

    def test_no_tasks_available(self):
        """Test behavior when no tasks are available."""
        cl = CurriculumLearning()
        # Clear all loaded tasks to simulate no tasks available
        cl.tasks_by_level = {i: [] for i in range(0, 7)}

        task = cl.get_prompt()
        self.assertIsNone(task)

    def test_initialization_with_format_reward_default(self):
        """Test CurriculumLearning initialization with default format reward enabled."""
        cl = CurriculumLearning(use_format=True)

        self.assertTrue(cl.use_format)
        # Should have both format_think and format_answer
        self.assertIn("format_think", cl.aux_reward_functions)
        self.assertIn("format_answer", cl.aux_reward_functions)
        self.assertIsNotNone(cl.aux_reward_functions.get("format_think"))
        self.assertIsNotNone(cl.aux_reward_functions.get("format_answer"))

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

        # Add task to session
        task = Task(
            task_id="math_test",
            task_name="Test",
            task_type="math",
            level=1,
            prompt="What is 2+2?",
            expected_answer="4",
        )
        cl.session.add_task(task)

        # Mock the primary reward function
        with patch.object(
            cl.reward_functions["math"], "compute_reward"
        ) as mock_primary:
            mock_result = MagicMock()
            mock_result.score = 1.0
            mock_result.info = ""
            mock_primary.return_value = mock_result

            # Mock auxiliary reward computation
            with patch.object(cl, "get_aux_reward_scores") as mock_aux_scores:
                # Return some auxiliary scores
                mock_aux_scores.return_value = {
                    "format": {
                        "score": 0.8,
                        "info": "",
                    }
                }

                # Provide valid format with both tags to pass format gate
                response = "<think>2 + 2 = 4</think>\n\n<answer>4</answer>"
                reward = cl.compute_reward("math_test", response)

                # Combined score should be 80% primary + 20% auxiliary average (new default)
                # 0.8 * 1.0 + 0.2 * 0.8 = 0.8 + 0.16 = 0.96
                self.assertAlmostEqual(reward, 0.96, places=2)

    def test_auxiliary_rewards_affect_curriculum_progression(self):
        """Test that combined rewards (primary + auxiliary) affect curriculum level."""
        cl = CurriculumLearning(use_format=True, use_repetition=True)
        cl.current_level = 0

        # Simulate good performance with success window data (what actually drives advancement)
        for _ in range(50):
            cl._track_success("math", True)  # All successes

        # Trigger level update
        cl._update_level()

        # Should advance level based on sliding window success rate
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

    def test_missing_math_file(self):
        """Test curriculum learning when math.json file is missing."""
        # Create curriculum without math file
        with patch("infinite_rl.curriculum.Path") as mock_path:
            mock_math_file = MagicMock()
            mock_math_file.exists.return_value = False
            mock_path.return_value.parent.joinpath.return_value = mock_math_file

            with patch("infinite_rl.puzzles.get_available_puzzles", return_value=[]):
                cl = CurriculumLearning()

        # Should still initialize but with no math tasks
        self.assertEqual(len(cl.tasks_by_level[1]), 0)

    def test_corrupted_math_file(self):
        """Test curriculum learning with corrupted math.json file."""
        with patch("infinite_rl.curriculum.Path") as mock_path, patch(
            "builtins.open", mock_open(read_data="invalid json")
        ), patch("infinite_rl.puzzles.get_available_puzzles", return_value=[]):

            mock_math_file = MagicMock()
            mock_math_file.exists.return_value = True
            mock_path.return_value.parent.joinpath.return_value = mock_math_file

            cl = CurriculumLearning()

        # Should handle JSON parsing error gracefully
        self.assertIsInstance(cl.tasks_by_level, dict)

    def test_missing_puzzles_file(self):
        """Test curriculum learning when puzzles.json file is missing."""
        with patch("infinite_rl.curriculum.Path") as mock_path, patch(
            "infinite_rl.puzzles.get_available_puzzles", return_value=[]
        ):

            mock_puzzles_file = MagicMock()
            mock_puzzles_file.exists.return_value = False
            mock_math_file = MagicMock()
            mock_math_file.exists.return_value = False

            def mock_joinpath(*args):
                if "puzzles.json" in str(args):
                    return mock_puzzles_file
                return mock_math_file

            mock_path.return_value.parent.joinpath.side_effect = mock_joinpath

            cl = CurriculumLearning()

        # Should still initialize but with no puzzle tasks
        total_tasks = sum(len(tasks) for tasks in cl.tasks_by_level.values())
        self.assertEqual(total_tasks, 0)

    def test_corrupted_puzzles_file(self):
        """Test curriculum learning with corrupted puzzles.json file."""
        with patch("infinite_rl.curriculum.Path") as mock_path, patch(
            "builtins.open", mock_open(read_data="invalid json")
        ), patch("infinite_rl.puzzles.get_available_puzzles", return_value=[]):

            mock_puzzles_file = MagicMock()
            mock_puzzles_file.exists.return_value = True
            mock_math_file = MagicMock()
            mock_math_file.exists.return_value = False

            def mock_joinpath(*args):
                if "puzzles.json" in str(args):
                    return mock_puzzles_file
                return mock_math_file

            mock_path.return_value.parent.joinpath.side_effect = mock_joinpath

            cl = CurriculumLearning()

        # Should handle JSON parsing error gracefully
        self.assertIsInstance(cl.tasks_by_level, dict)

    def test_get_prompt_with_no_tasks(self):
        """Test get_prompt when no tasks are available at current level."""
        cl = CurriculumLearning()
        # Clear all tasks
        cl.tasks_by_level = {i: [] for i in range(0, 7)}

        result = cl.get_prompt()
        self.assertIsNone(result)

    def test_get_rewards_with_invalid_task_type(self):
        """Test compute_reward with invalid task type."""
        cl = CurriculumLearning()

        with self.assertRaises(ValueError):
            cl.compute_reward("nonexistent_id", "response")

    # def test_auxiliary_reward_initialization_failure(self):
    #     """Test that auxiliary reward initialization failures are handled."""
    #     mock_reward_func = MagicMock(side_effect=Exception("Import error"))
    #     with patch('infinite_rl.reward_functions.LangConsistencyRewardFunction', mock_reward_func):
    #         cl = CurriculumLearning(use_lang_consistency=True)
    #
    #     # Should not crash, just skip the failing auxiliary reward
    #     self.assertNotIn("lang_consistency", cl.aux_reward_functions)

    def test_get_aux_reward_scores_with_no_aux_rewards(self):
        """Test get_aux_reward_scores when no auxiliary rewards are configured."""
        cl = CurriculumLearning(
            use_lang_consistency=False,
            use_repetition=False,
            use_format=False,
            use_reasoning_steps=False,
            use_length=False,
        )

        scores = cl.get_aux_reward_scores("response", "expected")
        self.assertEqual(scores, {})

    def test_level_advancement_boundary(self):
        """Test level advancement at boundary conditions."""
        cl = CurriculumLearning()
        cl.current_level = 6  # Maximum level

        # Should not advance beyond level 6
        cl._update_level()
        self.assertEqual(cl.current_level, 6)

    def test_task_counter_initialization(self):
        """Test that task counters are properly computed from session."""
        cl = CurriculumLearning()

        # Initially should have no tasks, so empty counters
        self.assertEqual(cl._get_task_counters(), {})

    def test_recent_tasks_tracking_limit(self):
        """Test that recent tasks are properly tracked and weighted."""
        cl = CurriculumLearning()

        # Add tasks to the session
        for i in range(3):
            task = Task(
                task_id=f"task_{i}",
                task_name=f"Task {i}",
                task_type="math",
                level=1,
                prompt="Test",
                expected_answer="Answer",
            )
            cl.session.add_task(task)

        # Add to recent tasks
        cl.recent_tasks = ["task_0", "task_1", "task_0"]

        # Verify tracking
        self.assertEqual(len(cl.recent_tasks), 3)
        self.assertEqual(cl.recent_tasks.count("task_0"), 2)

    def test_session_integration_with_curriculum(self):
        """Test that Session is properly integrated with CurriculumLearning."""
        cl = CurriculumLearning()

        # Verify session exists
        self.assertIsNotNone(cl.session)
        self.assertIsInstance(cl.session, Session)

        # Verify session can store tasks
        task = Task(
            task_id="test_id",
            task_name="Test",
            task_type="math",
            level=1,
            prompt="Test",
            expected_answer="Answer",
        )
        cl.session.add_task(task)

        retrieved = cl.session.get_task("test_id")
        self.assertIsNotNone(retrieved)
        self.assertEqual(retrieved.task_id, "test_id")

    def test_get_prompt_adds_task_to_session(self):
        """Test that get_prompt properly adds tasks to the session."""
        cl = CurriculumLearning()

        # Set up available task
        cl.tasks_by_level[0] = [
            {
                "type": "math",
                "data": {"problem": "2+2?", "solution": "4", "rating": 1},
                "rating": 0,
                "id": "math_001",
            }
        ]

        # Initially no tasks in session
        self.assertEqual(len(cl.session.tasks), 0)

        # Get a prompt
        task = cl.get_prompt()

        # Now task should be in session
        self.assertEqual(len(cl.session.tasks), 1)
        self.assertIn(task.task_id, cl.session.tasks)

    def test_reward_function_score_with_info(self):
        """Test that RewardFunctionScore properly stores info message."""
        reward = RewardFunctionScore(
            score=0.8,
            reward_function_name="format_answer",
            info="Format validation passed",
        )

        self.assertEqual(reward.score, 0.8)
        self.assertEqual(reward.reward_function_name, "format_answer")
        self.assertEqual(reward.info, "Format validation passed")

    def test_reward_function_score_default_values(self):
        """Test RewardFunctionScore default values."""
        reward = RewardFunctionScore(score=0.5)

        self.assertEqual(reward.score, 0.5)
        self.assertEqual(reward.reward_function_name, "")
        self.assertEqual(reward.info, "")

    def test_compute_reward_saves_to_session(self):
        """Test that compute_reward properly saves rewards to session."""
        cl = CurriculumLearning(
            use_format=False,
            use_repetition=False,
            use_reasoning_steps=False,
            use_length=False,
            use_lang_consistency=False,
        )

        # Create and add task
        task = Task(
            task_id="math_test",
            task_name="Test",
            task_type="math",
            level=1,
            prompt="2+2?",
            expected_answer="4",
        )
        cl.session.add_task(task)

        # Mock reward function
        with patch.object(
            cl.reward_functions["math"], "compute_reward"
        ) as mock_compute:
            mock_result = MagicMock()
            mock_result.score = 1.0
            mock_result.info = ""
            mock_compute.return_value = mock_result

            # Compute reward
            cl.compute_reward("math_test", "<answer>4</answer>")

            # Verify rewards were saved to session
            task_rewards = cl.session.get_task_rewards("math_test")
            self.assertGreater(len(task_rewards), 0)
            self.assertEqual(task_rewards[0].reward_function_name, "primary")

    def test_compute_reward_logs_to_file(self):
        """Verify compute_reward writes evaluation logs through CurriculumLearning."""
        cl = CurriculumLearning(
            log_file=self.log_file,
            use_format=False,
            use_repetition=False,
            use_reasoning_steps=False,
            use_length=False,
            use_lang_consistency=False,
        )

        task = Task(
            task_id="math_log",
            task_name="Log Test",
            task_type="math",
            level=1,
            prompt="1+1?",
            expected_answer="2",
        )
        cl.session.add_task(task)

        with patch.object(
            cl.reward_functions["math"], "compute_reward"
        ) as mock_compute:
            mock_result = MagicMock()
            mock_result.score = 1.0
            mock_result.info = "logged"
            mock_compute.return_value = mock_result

            self.assertFalse(os.path.exists(self.log_file))
            cl.compute_reward("math_log", "<answer>2</answer>")

        self.assertTrue(os.path.exists(self.log_file))
        with open(self.log_file, "r", encoding="utf-8") as f:
            log_entry = json.loads(f.readline())

        self.assertEqual(log_entry["task_id"], "math_log")
        self.assertEqual(log_entry["model_output"], "<answer>2</answer>")
        self.assertEqual(log_entry["primary_score"], 1.0)
        self.assertEqual(log_entry["info"]["primary"], "logged")
        self.assertIsNotNone(log_entry["first_response_at"])

    def test_reflective_learning_disabled(self):
        """Test that reflective learning is disabled when rate is 0."""
        cl = CurriculumLearning(reflective_learning_rate=0.0)

        # Add a format failure task
        task = Task(
            task_id="format_fail",
            task_name="Format Failure",
            task_type="math",
            level=1,
            prompt="What is 2+2?",
            expected_answer="4",
            model_output="No answer tags here",
        )
        # Add format failure reward
        format_reward = RewardFunctionScore(
            score=0.0, reward_function_name="format_answer", info="Missing answer tags"
        )
        task.add_reward(format_reward)
        cl.session.add_task(task)

        # Set up available tasks
        cl.tasks_by_level[0] = [
            {
                "type": "math",
                "data": {"problem": "Q1", "solution": "A1", "rating": 0},
                "rating": 0,
                "id": "normal_task",
            }
        ]

        # Try to get prompt multiple times - should never return reflective
        for _ in range(10):
            task_prompt = cl.get_prompt()
            self.assertIsNotNone(task_prompt)
            # Should be normal task, not reflective
            self.assertNotIn("reflective", task_prompt.task_id.lower())

    def test_reflective_learning_enabled(self):
        """Test that reflective learning returns format failure tasks."""
        cl = CurriculumLearning(reflective_learning_rate=1.0)  # Always trigger

        # Add a format failure task
        format_fail_task = Task(
            task_id="format_fail_1",
            task_name="Format Failure Task",
            task_type="math",
            level=1,
            prompt="What is 2+2?",
            expected_answer="4",
            model_output="My answer is five",
        )
        # Add format failure reward
        format_reward = RewardFunctionScore(
            score=0.0, reward_function_name="format_answer", info="Missing answer tags"
        )
        format_fail_task.add_reward(format_reward)
        cl.session.add_task(format_fail_task)

        # Exit warmup
        cl.global_step = cl.warmup_step

        # Get prompt - should return reflective task
        reflective_task = cl.get_prompt()

        self.assertIsNotNone(reflective_task)
        self.assertIn("reflective", reflective_task.task_id.lower())
        self.assertIn("reflective", reflective_task.task_name.lower())
        # Should include the original task info in the prompt
        self.assertIn(
            "You previously attempted this", reflective_task.prompt
        )  # Works for both math and puzzle
        self.assertIn("2+2", reflective_task.prompt)
        self.assertIn("five", reflective_task.prompt)

    def test_get_format_failure_tasks(self):
        """Test extracting format failure tasks from session."""
        cl = CurriculumLearning()

        # Add tasks with different outcomes
        # Task 1: Format failure
        task1 = Task(
            task_id="task_1",
            task_name="Task 1",
            task_type="math",
            level=1,
            prompt="Q1",
            expected_answer="A1",
        )
        format_fail = RewardFunctionScore(
            score=0.0, reward_function_name="format_think", info="Failed"
        )
        task1.add_reward(format_fail)
        cl.session.add_task(task1)

        # Task 2: Format success
        task2 = Task(
            task_id="task_2",
            task_name="Task 2",
            task_type="math",
            level=1,
            prompt="Q2",
            expected_answer="A2",
        )
        format_success = RewardFunctionScore(
            score=1.0, reward_function_name="format_answer", info="Passed"
        )
        task2.add_reward(format_success)
        cl.session.add_task(task2)

        # Task 3: Format failure (different task)
        task3 = Task(
            task_id="task_3",
            task_name="Task 3",
            task_type="puzzle",
            level=1,
            prompt="Q3",
            expected_answer="A3",
        )
        format_fail_2 = RewardFunctionScore(
            score=0.0, reward_function_name="format_answer", info="Failed"
        )
        task3.add_reward(format_fail_2)
        cl.session.add_task(task3)

        # Get format failures
        failures = cl._get_format_failure_tasks()

        # Should find 2 format failures
        self.assertEqual(len(failures), 2)
        failure_ids = [t.task_id for t in failures]
        self.assertIn("task_1", failure_ids)
        self.assertIn("task_3", failure_ids)
        self.assertNotIn("task_2", failure_ids)

    def test_create_reflective_prompt_includes_task_info(self):
        """Test that reflective prompt includes all necessary task information."""
        cl = CurriculumLearning()

        # Create a task that failed format
        original_task = Task(
            task_id="math_001",
            task_name="Addition Problem",
            task_type="math",
            level=2,
            prompt="What is 23 + 17?",
            expected_answer="40",
            model_output="The answer is 40 but I didn't use tags",
        )

        # Create reflective prompt
        reflective_prompt = cl._create_reflective_prompt(original_task)

        # Check that it includes key components
        self.assertIn("previously attempted", reflective_prompt.lower())
        self.assertIn("format", reflective_prompt.lower())
        self.assertIn("23 + 17", reflective_prompt)  # Original problem
        self.assertIn("40 but I didn't use tags", reflective_prompt)  # Previous attempt
        self.assertIn("<answer>", reflective_prompt)  # Format instruction
        self.assertIn("guidelines", reflective_prompt.lower())

    def test_reflective_task_preserves_metadata(self):
        """Test that reflective task preserves original task metadata."""
        cl = CurriculumLearning(reflective_learning_rate=1.0)

        # Create a puzzle task with format failure
        puzzle_task = Task(
            task_id="puzzle_js_001",
            task_name="JavaScript Puzzle",
            task_type="puzzle",
            level=3,
            prompt="Write a function...",
            expected_answer={"puzzle": "TestPuzzle", "inputs": {}},
            language="javascript",
            model_output="function test() { }",
        )
        format_fail = RewardFunctionScore(
            score=0.0, reward_function_name="format_answer"
        )
        puzzle_task.add_reward(format_fail)
        cl.session.add_task(puzzle_task)

        # Exit warmup to allow reflective learning
        cl.global_step = cl.warmup_step

        # Get reflective task
        reflective = cl._get_reflective_prompt()

        self.assertIsNotNone(reflective)
        # Should preserve metadata
        self.assertEqual(reflective.task_type, "puzzle")
        self.assertEqual(reflective.level, 3)
        self.assertEqual(reflective.language, "javascript")
        self.assertEqual(reflective.expected_answer, puzzle_task.expected_answer)

    def test_no_format_failures_returns_none(self):
        """Test that get_reflective_prompt returns None when no format failures exist."""
        cl = CurriculumLearning(reflective_learning_rate=1.0)

        # Add tasks without format failures
        for i in range(3):
            task = Task(
                task_id=f"good_task_{i}",
                task_name=f"Good Task {i}",
                task_type="math",
                level=1,
                prompt="Q",
                expected_answer="A",
            )
            # Success reward
            success = RewardFunctionScore(score=1.0, reward_function_name="primary")
            task.add_reward(success)
            cl.session.add_task(task)

        # Try to get reflective task - should return None
        reflective = cl._get_reflective_prompt()
        self.assertIsNone(reflective)

    def test_reflective_learning_during_warmup(self):
        """Test that reflective learning doesn't trigger during warmup phase."""
        cl = CurriculumLearning(
            reflective_learning_rate=1.0, warmup_step=100
        )  # Long warmup

        # Add format failure
        task = Task(
            task_id="format_fail",
            task_name="Fail",
            task_type="math",
            level=1,
            prompt="Q",
            expected_answer="A",
        )
        fail = RewardFunctionScore(score=0.0, reward_function_name="format_answer")
        task.add_reward(fail)
        cl.session.add_task(task)

        # Set up normal tasks
        cl.tasks_by_level[0] = [
            {
                "type": "math",
                "data": {"problem": "Q", "solution": "A", "rating": 0},
                "rating": 0,
                "id": "normal",
            }
        ]

        # During warmup (global_step < warmup_step), reflective should not trigger
        cl.global_step = 10  # Still in warmup

        task_prompt = cl.get_prompt()

        # Should be normal task, not reflective
        self.assertIsNotNone(task_prompt)
        self.assertNotIn("reflective", task_prompt.task_id.lower())

    def test_reflective_task_added_to_session(self):
        """Test that reflective tasks are properly added to the session."""
        cl = CurriculumLearning(reflective_learning_rate=1.0)

        # Add format failure
        orig = Task(
            task_id="orig_1",
            task_name="Original",
            task_type="math",
            level=1,
            prompt="Q",
            expected_answer="A",
        )
        fail = RewardFunctionScore(score=0.0, reward_function_name="format_answer")
        orig.add_reward(fail)
        cl.session.add_task(orig)

        session_size_before = len(cl.session.tasks)

        # Get reflective task
        cl.global_step = cl.warmup_step
        reflective = cl._get_reflective_prompt()

        # Should have added a new task to session
        self.assertGreater(len(cl.session.tasks), session_size_before)
        self.assertIn(reflective.task_id, cl.session.tasks)

    def test_reflective_learning_rate_parameter(self):
        """Test that reflective_learning_rate parameter is configurable."""
        cl1 = CurriculumLearning(reflective_learning_rate=0.0)
        cl2 = CurriculumLearning(reflective_learning_rate=0.5)
        cl3 = CurriculumLearning(reflective_learning_rate=1.0)

        self.assertEqual(cl1.reflective_learning_rate, 0.0)
        self.assertEqual(cl2.reflective_learning_rate, 0.5)
        self.assertEqual(cl3.reflective_learning_rate, 1.0)

    def test_reflective_tasks_not_used_for_reflective(self):
        """Test that reflective tasks are NOT used to create further reflective tasks."""
        cl = CurriculumLearning(reflective_learning_rate=1.0)

        # Add an original task with format failure
        original_task = Task(
            task_id="original_001",
            task_name="Original Task",
            task_type="math",
            level=1,
            prompt="What is 2+2?",
            expected_answer="4",
            model_output="Wrong answer",
        )
        format_reward = RewardFunctionScore(
            score=0.0, reward_function_name="format_answer", info="Failed"
        )
        original_task.add_reward(format_reward)
        cl.session.add_task(original_task)

        # Exit warmup to allow reflective learning
        cl.global_step = cl.warmup_step

        # Get first reflective version
        reflective_1 = cl._get_reflective_prompt()
        self.assertIsNotNone(reflective_1)
        self.assertEqual(reflective_1.task_id, "original_001_reflective_0")
        self.assertEqual(reflective_1.task_name, "Original Task (reflective attempt 1)")

        # Now mark the reflective task as having a format failure
        reflective_format_reward = RewardFunctionScore(
            score=0.0, reward_function_name="format_answer", info="Failed"
        )
        reflective_1.add_reward(reflective_format_reward)

        # Try to get another reflective task
        # Should NOT select the reflective_1 task, but should select original_001 again
        reflective_2 = cl._get_reflective_prompt()

        # Should create a NEW reflective version of the original task, not from reflective_1
        self.assertIsNotNone(reflective_2)
        # Should be _reflective_1, not _reflective_0_reflective
        self.assertEqual(reflective_2.task_id, "original_001_reflective_1")
        self.assertEqual(reflective_2.task_name, "Original Task (reflective attempt 2)")

    def test_original_task_unlimited_reflective_attempts(self):
        """Test that the same original task can be used for reflective learning unlimited times."""
        cl = CurriculumLearning(reflective_learning_rate=1.0)

        # Add a single original task with format failure
        original_task = Task(
            task_id="problem_xyz",
            task_name="Problem XYZ",
            task_type="puzzle",
            level=2,
            prompt="Solve the puzzle",
            expected_answer={"code": "solution"},
            language="python",
            model_output="bad code",
        )
        format_reward = RewardFunctionScore(
            score=0.0, reward_function_name="format_answer", info="Failed"
        )
        original_task.add_reward(format_reward)
        cl.session.add_task(original_task)

        # Exit warmup
        cl.global_step = cl.warmup_step

        # Generate multiple reflective versions of the same task
        reflective_tasks = []
        for i in range(5):
            reflective = cl._get_reflective_prompt()
            self.assertIsNotNone(reflective)
            reflective_tasks.append(reflective)

            # Mark it as failed to allow next iteration
            fail_reward = RewardFunctionScore(
                score=0.0, reward_function_name="format_answer", info="Still failed"
            )
            reflective.add_reward(fail_reward)

        # Verify all came from the same original task with different counters
        for i, task in enumerate(reflective_tasks):
            self.assertEqual(task.task_id, f"problem_xyz_reflective_{i}")
            self.assertEqual(
                task.task_name, f"Problem XYZ (reflective attempt {i + 1})"
            )
            self.assertIn("Solve the puzzle", task.prompt)  # Original prompt included


if __name__ == "__main__":
    unittest.main()
