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


class MockTokenizer:
    """Mock tokenizer for testing without transformers dependency."""

    def __init__(self):
        self.bos_token = "<BOS>"

    def apply_chat_template(self, conversation, tokenize=False):
        """Mock chat template application."""
        parts = []
        for msg in conversation:
            role = msg.get("role", "").upper()
            content = msg.get("content", "")
            parts.append(f"[{role}] {content}")

        result = " ".join(parts)
        return self.bos_token + result


def setup_llm_judge_with_mock_tokenizer(curriculum_learning):
    """Helper to set up LLM Judge with mock tokenizer in tests."""
    if "llm_judge" in curriculum_learning.aux_reward_functions:
        curriculum_learning.aux_reward_functions["llm_judge"].tokenizer = (
            MockTokenizer()
        )


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
        self.assertEqual(len(self.task.generations), 0)
        self.assertIsNone(self.task.model_output)
        self.assertIsNotNone(self.task.created_at)
        self.assertIsNone(self.task.first_response_at)

    def test_add_generation(self):
        """Test adding rewards to a task."""
        rewards = [
            RewardFunctionScore(
                score=1.0,
                reward_function_name="primary",
                info="",
            )
        ]
        self.task.add_generation("test output", rewards, 1.0)

        self.assertEqual(len(self.task.generations), 1)
        self.assertEqual(len(self.task.generations[0].rewards), 1)
        self.assertEqual(self.task.generations[0].rewards[0].score, 1.0)
        self.assertEqual(
            self.task.generations[0].rewards[0].reward_function_name, "primary"
        )

    def test_get_score(self):
        """Test getting the primary score from a task."""
        # No generations yet
        self.assertEqual(self.task.get_score(), 0.0)

        # Add generation with primary reward
        rewards = [
            RewardFunctionScore(
                score=0.8,
                reward_function_name="primary",
                info="",
            )
        ]
        self.task.add_generation("test output", rewards, 0.8)

        # Should return primary score from latest generation
        self.assertEqual(self.task.get_score(), 0.8)

        # Add another generation with auxiliary reward (should not affect get_score)
        aux_rewards = [
            RewardFunctionScore(
                score=0.8,
                reward_function_name="primary",
                info="",
            ),
            RewardFunctionScore(
                score=0.9,
                reward_function_name="format_answer",
                info="",
            ),
        ]
        self.task.add_generation("test output 2", aux_rewards, 0.8)

        # Should still return primary score from latest generation
        self.assertEqual(self.task.get_score(), 0.8)

    def test_task_to_dict(self):
        """Test converting task to dictionary."""
        rewards = [
            RewardFunctionScore(
                score=1.0,
                reward_function_name="primary",
                info="",
            )
        ]
        self.task.add_generation("test output", rewards, 1.0)

        task_dict = self.task.to_dict()

        self.assertEqual(task_dict["task_id"], "math_001")
        self.assertEqual(task_dict["task_name"], "Simple Addition")
        self.assertEqual(task_dict["task_type"], "math")
        self.assertEqual(task_dict["level"], 1)
        self.assertEqual(len(task_dict["generations"]), 1)
        self.assertEqual(len(task_dict["generations"][0]["rewards"]), 1)
        self.assertEqual(task_dict["generations"][0]["rewards"][0]["score"], 1.0)
        # model_output is in generation, not at task level
        self.assertEqual(task_dict["generations"][0]["output"], "test output")
        self.assertIsNotNone(task_dict["created_at"])
        self.assertIsNotNone(task_dict["first_response_at"])

    def test_task_generation_tracking(self):
        """Test that Task properly tracks generations."""
        # Test initial state
        self.assertEqual(len(self.task.generations), 0)
        self.assertIsNone(self.task.latest_generation)

        # Add first generation
        reward1 = RewardFunctionScore(0.8, "primary", "good answer")
        gen1 = self.task.add_generation("output1", [reward1], 0.8)

        self.assertEqual(len(self.task.generations), 1)
        self.assertEqual(self.task.generations[0].output, "output1")
        self.assertEqual(self.task.generations[0].primary_score, 0.8)
        self.assertEqual(self.task.latest_generation, gen1)

        # Add second generation
        reward2 = RewardFunctionScore(0.6, "primary", "okay answer")
        gen2 = self.task.add_generation("output2", [reward2], 0.6)

        self.assertEqual(len(self.task.generations), 2)
        self.assertEqual(self.task.generations[1].output, "output2")
        self.assertEqual(self.task.latest_generation, gen2)

        # Test latest generation properties
        self.assertEqual(self.task.latest_generation.output, "output2")
        self.assertEqual(len(self.task.latest_generation.rewards), 1)
        self.assertEqual(self.task.latest_generation.rewards[0].score, 0.6)
        self.assertEqual(self.task.latest_generation.is_correct, True)  # 0.6 >= 0.5

        # Test to_dict includes generations
        task_dict = self.task.to_dict()
        self.assertIn("generations", task_dict)
        self.assertEqual(len(task_dict["generations"]), 2)
        self.assertEqual(task_dict["generations"][0]["output"], "output1")
        self.assertEqual(task_dict["generations"][1]["output"], "output2")


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

    def test_get_batch_data(self):
        """Test retrieving all generations data for a task."""
        task = Task(
            task_id="task_001",
            task_name="Test Task",
            task_type="math",
            level=1,
            prompt="Test prompt",
            expected_answer="4",
        )

        # Add multiple generations to simulate GRPO batch
        rewards1 = [
            RewardFunctionScore(score=0.5, reward_function_name="primary", info="")
        ]
        rewards2 = [
            RewardFunctionScore(score=0.8, reward_function_name="primary", info="")
        ]
        rewards3 = [
            RewardFunctionScore(score=1.0, reward_function_name="primary", info="")
        ]

        task.add_generation("First attempt", rewards1, 0.5)
        task.add_generation("Second attempt", rewards2, 0.8)
        task.add_generation("Third attempt", rewards3, 1.0)

        self.session.add_task(task)

        batch_data = self.session.get_batch_data("task_001")

        self.assertIsNotNone(batch_data)
        self.assertEqual(len(batch_data), 3)

        # Check first generation
        self.assertEqual(batch_data[0]["output"], "First attempt")
        self.assertEqual(batch_data[0]["primary_score"], 0.5)
        self.assertTrue(batch_data[0]["is_correct"])  # 0.5 >= 0.5
        self.assertEqual(len(batch_data[0]["rewards"]), 1)

        # Check third generation (correct)
        self.assertEqual(batch_data[2]["output"], "Third attempt")
        self.assertEqual(batch_data[2]["primary_score"], 1.0)
        self.assertTrue(batch_data[2]["is_correct"])

    def test_get_batch_data_nonexistent_task(self):
        """Test get_batch_data returns None for non-existent task."""
        batch_data = self.session.get_batch_data("nonexistent")
        self.assertIsNone(batch_data)

    def test_get_batch_stats(self):
        """Test retrieving statistics about generations for a task."""
        task = Task(
            task_id="task_001",
            task_name="Test Task",
            task_type="math",
            level=1,
            prompt="Test prompt",
            expected_answer="4",
        )

        # Add generations with varying scores
        task.add_generation(
            "Bad",
            [RewardFunctionScore(score=0.2, reward_function_name="primary", info="")],
            0.2,
        )
        task.add_generation(
            "Good",
            [RewardFunctionScore(score=0.9, reward_function_name="primary", info="")],
            0.9,
        )
        task.add_generation(
            "Perfect",
            [RewardFunctionScore(score=1.0, reward_function_name="primary", info="")],
            1.0,
        )

        self.session.add_task(task)

        stats = self.session.get_batch_stats("task_001")

        self.assertIsNotNone(stats)
        self.assertEqual(stats["num_generations"], 3)
        self.assertEqual(stats["scores"]["min"], 0.2)
        self.assertEqual(stats["scores"]["max"], 1.0)
        self.assertAlmostEqual(stats["scores"]["avg"], 0.7, places=1)
        self.assertEqual(stats["best_generation"]["index"], 2)
        self.assertEqual(stats["best_generation"]["score"], 1.0)
        self.assertEqual(stats["best_generation"]["output"], "Perfect")
        self.assertEqual(
            stats["correct_generations"], 2
        )  # Scores 0.9 and 1.0 are >= 0.5
        self.assertEqual(
            stats["first_correct_at"], 1
        )  # First generation (score 0.2) is not correct, second (0.9) is

    def test_get_batch_stats_no_generations(self):
        """Test get_batch_stats returns None for task with no generations."""
        task = Task(
            task_id="task_001",
            task_name="Test Task",
            task_type="math",
            level=1,
            prompt="Test prompt",
            expected_answer="4",
        )
        self.session.add_task(task)

        stats = self.session.get_batch_stats("task_001")
        self.assertIsNone(stats)

    def test_get_batch_stats_single_generation(self):
        """Test get_batch_stats works with single generation."""
        task = Task(
            task_id="task_001",
            task_name="Test Task",
            task_type="math",
            level=1,
            prompt="Test prompt",
            expected_answer="4",
        )

        task.add_generation(
            "Only attempt",
            [RewardFunctionScore(score=0.7, reward_function_name="primary", info="")],
            0.7,
        )
        self.session.add_task(task)

        stats = self.session.get_batch_stats("task_001")

        self.assertIsNotNone(stats)
        self.assertEqual(stats["num_generations"], 1)
        self.assertEqual(stats["scores"]["min"], 0.7)
        self.assertEqual(stats["scores"]["max"], 0.7)
        self.assertEqual(stats["scores"]["avg"], 0.7)
        self.assertEqual(stats["scores"]["std"], 0.0)  # No variance with single value
        self.assertEqual(stats["correct_generations"], 1)  # 0.7 >= 0.5
        self.assertEqual(
            stats["first_correct_at"], 0
        )  # First (only) generation is correct

    def test_create_math_task(self):
        """Test creating a math task."""
        session = Session()

        task_data = {
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

        task = session.create_math_task(task_data)

        self.assertIsNotNone(task)
        self.assertIsInstance(task, Task)
        self.assertEqual(task.task_type, "math")
        self.assertEqual(task.level, 1)
        self.assertEqual(task.expected_answer, "4")
        self.assertEqual(task.language, "en")
        self.assertIn("What is 2+2?", task.prompt)
        self.assertIn(task.task_id, session.tasks)

    def test_create_puzzle_task(self):
        """Test creating a puzzle task."""
        session = Session()

        task_data = {
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

        task = session.create_puzzle_task(task_data)

        self.assertIsNotNone(task)
        self.assertIsInstance(task, Task)
        self.assertEqual(task.task_type, "puzzle")
        self.assertEqual(task.level, 1)
        self.assertEqual(task.language, "javascript")
        self.assertIn("TestPuzzle", task.prompt)
        self.assertIn("Test puzzle description", task.prompt)
        self.assertIn(task.task_id, session.tasks)

    def test_create_truthy_task(self):
        """Test creating a truthy task."""
        session = Session()

        task_data = {
            "type": "truthy",
            "data": {
                "prompt": "Test prompt",
                "chosen": "Good answer",
                "rejected": "Bad answer",
                "id": "truthy_test",
                "language": "en",
            },
            "rating": None,
            "id": "truthy_test",
        }

        task = session.create_truthy_task(task_data)

        self.assertIsNotNone(task)
        self.assertIsInstance(task, Task)
        self.assertEqual(task.task_type, "truthy")
        self.assertEqual(task.level, -1)
        self.assertIn("Test prompt", task.prompt)
        self.assertIn(task.task_id, session.tasks)

    def test_task_instance_counter(self):
        """Test that task instance counter increments properly."""
        session = Session()
        initial_counter = session.task_instance_counter

        # Create tasks
        math_data = {
            "type": "math",
            "data": {"prompt": "Test", "response": "4", "lang": "en"},
            "rating": 1,
            "id": "math_1",
        }
        puzzle_data = {
            "type": "puzzle",
            "language": "python",
            "puzzle_name": "Test",
            "data": {
                "name": "Test",
                "docstring": "",
                "sat": "def sat(): return True",
                "sol": "def sol(): pass",
                "ans_type": "boolean",
            },
            "rating": 1,
            "id": "puzzle_1",
        }

        math_task = session.create_math_task(math_data)
        self.assertEqual(session.task_instance_counter, initial_counter + 1)
        self.assertIn(str(initial_counter), math_task.task_id)

        puzzle_task = session.create_puzzle_task(puzzle_data)
        self.assertEqual(session.task_instance_counter, initial_counter + 2)
        self.assertIn(str(initial_counter + 1), puzzle_task.task_id)

    def test_get_recent_task_ids(self):
        """Test getting recent task base IDs."""
        session = Session()

        # Add tasks to history
        session.task_history = ["math_1_0", "puzzle_2_1", "math_1_2", "truthy_3_3"]

        recent_ids = session._get_recent_task_ids()

        # Should extract base IDs
        expected = ["math_1", "puzzle_2", "math_1", "truthy_3"]
        self.assertEqual(recent_ids, expected)


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
                    self.assertEqual(len(cl.session._get_recent_task_ids()), 0)

    def test_get_rewards_correct_answer(self):
        """Test get_rewards with correct answer."""
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
            combined_score = cl.compute_reward("math_test", response)

            # compute_reward returns combined_score (includes judge score when batch complete)
            # Since this is first generation, combined_score should be initialized after batch is complete
            self.assertIsNotNone(combined_score)
            self.assertGreaterEqual(combined_score, 0.0)
            self.assertLessEqual(combined_score, 1.0)

            # Check that task has is_correct set to True
            task_from_session = cl.session.get_task("math_test")
            self.assertTrue(task_from_session.is_correct)

    def test_get_rewards_wrong_answer(self):
        """Test get_rewards with wrong answer."""
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

            combined_score = cl.compute_reward("math_test", "<answer>5</answer>")

            # compute_reward returns combined_score (will be 0.0 or low since answer is wrong)
            self.assertIsNotNone(combined_score)
            self.assertLessEqual(combined_score, 1.0)

            # Check that task has is_correct set to False
            task_from_session = cl.session.get_task("math_test")
            self.assertFalse(task_from_session.is_correct)

    def test_get_rewards_truthy_uses_judge_as_primary(self):
        """Test get_rewards uses LLM Judge score as primary for truthy tasks."""
        cl = CurriculumLearning(
            use_llm_judge=True,
            llm_judge_kwargs={
                "api_host": "localhost",
                "api_port": 8000,
                "model_name": "test-model",  # Use a dummy name
            },
        )

        # Mock the judge function to avoid initialization issues
        mock_judge = MagicMock()
        cl.aux_reward_functions["llm_judge"] = mock_judge

        # Add a truthy task to the session
        task = Task(
            task_id="truthy_test",
            task_name="Truthy Test",
            task_type="truthy",
            level=0,
            prompt="Which is better?",
            expected_answer={"type": "truthy", "conversation": []},
        )
        cl.session.add_task(task)

        # Provide response
        response = "My response"
        combined_score = cl.compute_reward("truthy_test", response)

        # compute_reward returns placeholder combined_score (0.0 initially, batch not complete)
        self.assertEqual(combined_score, 0.0)

        # Check that task has generations with rewards
        task_from_session = cl.session.get_task("truthy_test")
        self.assertEqual(len(task_from_session.generations), 1)

    def test_level_advancement(self):
        """Test automatic level advancement."""
        cl = CurriculumLearning()
        cl.current_level = 0

        # Simulate good performance by adding successful tasks to the session
        # Need at least 10 samples at current level (0) due to min_samples constraint
        for i in range(10):
            task = Task(
                task_id=f"math_{i}",
                task_name=f"Math Task {i}",
                task_type="math",
                level=0,  # Must be at current level to be tracked
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
                level=0,  # Must be at current level to be tracked
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
        cl.session.tasks_by_level[0] = [
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

        # Mock random.random to avoid truthy task selection (20% chance)
        with patch("random.random", return_value=0.5):
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
        cl.session.tasks_by_level[0] = [
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

        # Mock random.random to avoid truthy task selection
        with patch("random.random", return_value=0.5):
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
        cl.session.tasks_by_level[0] = [
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
        # Mock random to avoid truthy selection
        with patch("random.random", return_value=0.5):
            task = cl.get_prompt()

        self.assertIsNotNone(task)
        # Task should have math_taskX_0 format with instance counter
        self.assertIn("math_task", task.task_id)

    def test_unique_task_ids_generated(self):
        """Test that each task instance gets a unique task_id."""
        cl = CurriculumLearning()

        # Set up a single task at level 0
        cl.session.tasks_by_level[0] = [
            {
                "type": "math",
                "data": {"prompt": "Q1", "response": "A1", "lang": "en", "rating": 1},
                "rating": 1,
                "id": "math_simple",
            },
        ]

        # Get multiple prompts from the same task - each should have unique ID
        # Mock random to avoid truthy selection and ensure consistent selection
        with patch("random.random", return_value=0.5), patch(
            "random.choices", return_value=[cl.session.tasks_by_level[0][0]]
        ):
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
        self.assertEqual(cl.session.task_instance_counter, 3)

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
        self.assertIsInstance(stats["truthy_tasks_count"], int)

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
        """Test behavior when no tasks are available at current level."""
        cl = CurriculumLearning()
        # Clear tasks at current level (level 0) to simulate no tasks available at that level
        cl.session.tasks_by_level[0] = []
        cl.session.truthy_tasks = []

        # Mock random to avoid truthy selection
        with patch("random.random", return_value=0.5):
            task = cl.get_prompt()

        # Should fallback to other levels or return None
        # Since session loads tasks, it should find tasks at other levels
        self.assertIsNotNone(task)

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
        # Note: lang_consistency may not be in aux_reward_functions if cantofilter dependency is missing
        # This is expected behavior - it logs a warning but doesn't fail initialization

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

        # Create a task with expected answer
        task = Task(
            task_id="test_task",
            task_name="Test Task",
            task_type="math",
            level=0,
            prompt="What is 2+2?",
            expected_answer="4",
        )
        task.model_output = "<think>This is my reasoning</think>\n<answer>4</answer>"

        # Get auxiliary scores
        aux_scores = cl.get_aux_reward_scores(task, is_correct=True)

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
                combined_score = cl.compute_reward("math_test", response)

                # compute_reward returns combined_score directly
                # Aux score 0.8 is normalized: clipped to [-1, 1] = 0.8, then (0.8 + 1) / 2 = 0.9
                # Since only one auxiliary, aux_avg = 0.9
                # Combined: 0.9 * 1.0 + 0.1 * 0.9 = 0.99 (with aux_weight=0.1)
                self.assertIsNotNone(combined_score)
                self.assertGreaterEqual(combined_score, 0.0)
        cl.current_level = 0

        # Simulate good performance with success window data (what actually drives advancement)
        # Track at current level (0) using GRPO batch-level tracking (primary_scores)
        for _ in range(50):
            cl._track_success_group(0, [1.0])  # Batch with perfect score (1.0)

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

        # Create a task
        task = Task(
            task_id="test_task",
            task_name="Test Task",
            task_type="math",
            level=0,
            prompt="What is 2+2?",
            expected_answer="4",
        )
        task.model_output = "<answer>4</answer>"

        aux_scores = cl.get_aux_reward_scores(task, is_correct=True)

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
        task = Task(
            task_id="test_task",
            task_name="Test Task",
            task_type="math",
            level=0,
            prompt="What is 2+2?",
            expected_answer="4",
        )
        task.model_output = "<think>Reasoning here</think>\n<answer>4</answer>"
        aux_scores = cl.get_aux_reward_scores(task, is_correct=True)
        self.assertIsInstance(aux_scores, dict)

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
        self.assertIsInstance(cl.session.tasks_by_level, dict)

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
        self.assertIsInstance(cl.session.tasks_by_level, dict)

    def test_get_prompt_with_no_tasks(self):
        """Test get_prompt when no tasks are available at current level."""
        cl = CurriculumLearning()
        # Clear tasks at current level (level 0)
        original_level_0 = cl.session.tasks_by_level[0][:]
        cl.session.tasks_by_level[0] = []
        cl.session.truthy_tasks = []

        # Mock random to avoid truthy selection
        with patch("random.random", return_value=0.5):
            result = cl.get_prompt()

        # Should fallback to other levels that have tasks
        self.assertIsNotNone(result)

        # Restore for other tests
        cl.session.tasks_by_level[0] = original_level_0

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
            use_whitespace_collapse=False,
        )

        # Create a simple task object for testing
        task = Task(
            task_id="test_1",
            task_name="test",
            task_type="math",
            level=0,
            prompt="test",
            expected_answer="42",
        )
        task.model_output = "response"

        scores = cl.get_aux_reward_scores(task, is_correct=True)
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
        cl.session.tasks_by_level[0] = [
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
            primary_score = cl.compute_reward("math_test", "<answer>4</answer>")

            # Verify primary score is returned
            self.assertEqual(primary_score, 1.0)

            # Verify rewards were saved to task
            task_from_session = cl.session.get_task("math_test")
            self.assertEqual(len(task_from_session.generations), 1)
            rewards = task_from_session.generations[0].rewards
            self.assertGreater(len(rewards), 0)
            self.assertEqual(rewards[0].reward_function_name, "primary")
            self.assertEqual(rewards[0].score, 1.0)

    def test_compute_reward_logs_to_file(self):
        """Verify compute_reward writes evaluation logs through CurriculumLearning."""
        cl = CurriculumLearning(
            log_file=self.log_file,
            use_format=False,
            use_repetition=False,
            use_reasoning_steps=False,
            use_length=False,
            use_lang_consistency=False,
            num_generations=1,
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

            # Logging should happen automatically when batch is complete (num_generations=1)

        self.assertTrue(os.path.exists(self.log_file))
        with open(self.log_file, "r", encoding="utf-8") as f:
            log_entry = json.loads(f.readline())

        self.assertEqual(log_entry["task_id"], "math_log")
        # model_output, primary_score, and info are now in generations array
        self.assertEqual(log_entry["generations"][0]["output"], "<answer>2</answer>")
        self.assertEqual(log_entry["generations"][0]["primary_score"], 1.0)
        # Find primary reward in generation
        primary_info = None
        for reward in log_entry["generations"][0]["rewards"]:
            if reward["reward_function_name"] == "primary":
                primary_info = reward["info"]
                break
        self.assertEqual(primary_info, "logged")

    def test_log_completed_task(self):
        """Test _log_completed_task logs task details correctly."""
        cl = CurriculumLearning(
            log_file=self.log_file,
            use_format=False,
            use_repetition=False,
            use_reasoning_steps=False,
            use_length=False,
            use_lang_consistency=False,
            num_generations=2,
        )

        # Create a task with rewards and generations
        task = Task(
            task_id="test_task",
            task_name="Test Task",
            task_type="math",
            level=1,
            prompt="2+2?",
            expected_answer="4",
        )
        cl.session.add_task(task)

        # Add some rewards
        primary_reward = RewardFunctionScore(
            score=1.0,
            reward_function_name="primary",
            info="Correct answer",
        )
        aux_reward = RewardFunctionScore(
            score=0.8,
            reward_function_name="format_answer",
            info="Good format",
        )
        task.add_generation("test output", [primary_reward, aux_reward], 1.0)

        # Add generations
        task.add_generation("Response 1", [primary_reward, aux_reward], 1.0)
        task.add_generation("Response 2", [primary_reward, aux_reward], 1.0)

        # Call _log_completed_task
        cl._log_completed_task(task)

        # Check log file
        self.assertTrue(os.path.exists(self.log_file))
        with open(self.log_file, "r", encoding="utf-8") as f:
            log_entry = json.loads(f.readline())

        # Verify log entry contents (task-level metadata)
        self.assertEqual(log_entry["task_id"], "test_task")
        self.assertEqual(log_entry["task_name"], "Test Task")
        self.assertEqual(log_entry["task_type"], "math")
        self.assertEqual(log_entry["level"], 1)
        self.assertEqual(log_entry["prompt"], "2+2?")
        # expected_answer is in generation, not at task level
        self.assertIn("generations", log_entry)
        self.assertEqual(len(log_entry["generations"]), 3)
        # Check latest generation has the scores and rewards
        latest_gen = log_entry["generations"][-1]
        self.assertEqual(latest_gen["primary_score"], 1.0)
        # Find primary and format_answer rewards
        primary_info = None
        format_score = None
        format_info = None
        for reward in latest_gen["rewards"]:
            if reward["reward_function_name"] == "primary":
                primary_info = reward["info"]
            elif reward["reward_function_name"] == "format_answer":
                format_score = reward["score"]
                format_info = reward["info"]
        self.assertEqual(primary_info, "Correct answer")
        self.assertEqual(format_score, 0.8)
        self.assertEqual(format_info, "Good format")
        self.assertIn("timestamp", log_entry)

    def test_log_completed_task_truthy(self):
        """Test _log_completed_task handles truthy tasks with judge score override."""
        cl = CurriculumLearning(
            log_file=self.log_file,
            use_llm_judge=True,
            llm_judge_kwargs={
                "api_host": "localhost",
                "api_port": 8000,
                "model_name": "test_model",
            },
            num_generations=1,
        )

        # Create a truthy task
        task = Task(
            task_id="truthy_task",
            task_name="Truthy Task",
            task_type="truthy",
            level=0,
            prompt="Which is better?",
            expected_answer={"chosen": "A", "rejected": "B"},
        )
        cl.session.add_task(task)

        # Add rewards: primary placeholder and llm_judge
        primary_reward = RewardFunctionScore(
            score=0.5,
            reward_function_name="primary",
            info="Placeholder",
        )
        judge_reward = RewardFunctionScore(
            score=0.9,
            reward_function_name="llm_judge",
            info="Good quality",
        )
        # Add generation
        task.add_generation("Response", [primary_reward, judge_reward], 0.9)

        # Call _log_completed_task
        cl._log_completed_task(task)

        # Check log file
        self.assertTrue(os.path.exists(self.log_file))
        with open(self.log_file, "r", encoding="utf-8") as f:
            log_entry = json.loads(f.readline())

        # For truthy tasks, primary_score should be the judge score (in generation)
        self.assertEqual(log_entry["task_type"], "truthy")
        latest_gen = log_entry["generations"][0]
        self.assertEqual(
            latest_gen["primary_score"], 0.9
        )  # Primary is judge score for truthy
        # Check rewards
        primary_info = None
        judge_score = None
        judge_info = None
        for reward in latest_gen["rewards"]:
            if reward["reward_function_name"] == "primary":
                primary_info = reward["info"]
            elif reward["reward_function_name"] == "llm_judge":
                judge_score = reward["score"]
                judge_info = reward["info"]
        self.assertEqual(primary_info, "Placeholder")  # Original primary info
        self.assertEqual(judge_score, 0.9)
        self.assertEqual(judge_info, "Good quality")

    def test_aux_weight_default_value(self):
        """Test that aux_weight defaults to 0.2."""
        cl = CurriculumLearning()
        self.assertEqual(cl.aux_weight, 0.2)

    def test_llm_judge_weight_default_value(self):
        """Test that llm_judge_weight defaults to 0.2."""
        cl = CurriculumLearning()
        self.assertEqual(cl.llm_judge_weight, 0.2)

    def test_custom_aux_weight(self):
        """Test setting custom aux_weight."""
        cl = CurriculumLearning(aux_weight=0.15)
        self.assertEqual(cl.aux_weight, 0.15)

    def test_custom_llm_judge_weight(self):
        """Test setting custom llm_judge_weight."""
        cl = CurriculumLearning(llm_judge_weight=0.3)
        self.assertEqual(cl.llm_judge_weight, 0.3)

    def test_llm_judge_requires_api_host(self):
        """Test that use_llm_judge=True requires api_host."""
        with self.assertRaises(ValueError) as context:
            CurriculumLearning(
                use_llm_judge=True,
                llm_judge_kwargs={"api_port": 8000},
            )
        self.assertIn("api_host", str(context.exception))

    def test_llm_judge_requires_api_port(self):
        """Test that use_llm_judge=True requires api_port."""
        with self.assertRaises(ValueError) as context:
            CurriculumLearning(
                use_llm_judge=True,
                llm_judge_kwargs={"api_host": "localhost"},
            )
        self.assertIn("api_port", str(context.exception))

    def test_llm_judge_initialization_with_valid_config(self):
        """Test that use_llm_judge=True works with api_host and api_port."""
        with patch.object(
            CurriculumLearning, "_initialize_aux_reward_functions"
        ) as mock_init:
            cl = CurriculumLearning(
                use_llm_judge=True,
                llm_judge_kwargs={
                    "api_host": "localhost",
                    "api_port": 8000,
                    "model_name": "Skywork",
                },
            )
            self.assertTrue(cl.use_llm_judge)
            self.assertEqual(cl.llm_judge_kwargs["api_host"], "localhost")
            self.assertEqual(cl.llm_judge_kwargs["api_port"], 8000)

    def test_llm_judge_disabled_by_default(self):
        """Test that use_llm_judge is disabled by default."""
        cl = CurriculumLearning()
        self.assertFalse(cl.use_llm_judge)

    def test_llm_judge_kwargs_empty_by_default(self):
        """Test that llm_judge_kwargs defaults to empty dict."""
        cl = CurriculumLearning()
        self.assertEqual(cl.llm_judge_kwargs, {})

    def test_truthy_task_requires_llm_judge(self):
        """Test that truthy tasks require llm_judge to be enabled."""
        cl = CurriculumLearning(use_llm_judge=False)

        # Create a truthy task manually
        task = Task(
            task_id="truthy_test",
            task_name="Truthy Test",
            task_type="truthy",
            level=0,
            prompt="Which response is better?",
            expected_answer={"type": "truthy", "conversation": []},
        )
        cl.session.add_task(task)

        # compute_reward should raise error since llm_judge is disabled
        with self.assertRaises(ValueError) as context:
            cl.compute_reward("truthy_test", "Test response")
        self.assertIn("llm_judge", str(context.exception))

    def test_truthy_task_primary_score_from_llm_judge(self):
        """Test that truthy task primary score comes from llm_judge via batch processing."""
        cl = CurriculumLearning(
            use_llm_judge=True,
            aux_weight=0.0,  # Disable auxiliary weight blending for clean test
            num_generations=1,  # Single generation mode for immediate processing
            llm_judge_kwargs={
                "api_host": "localhost",
                "api_port": 8000,
                "model_name": "Skywork/Skywork-Reward-V2-Qwen3-4B",
            },
            use_format=False,  # Disable format to avoid auxiliary reward blending
            use_whitespace_collapse=False,  # Disable whitespace collapse
            use_lang_consistency=False,  # Disable auxiliary rewards to test just LLM Judge
            use_repetition=False,
        )

        # Set up mock tokenizer for LLM Judge
        setup_llm_judge_with_mock_tokenizer(cl)

        # Create a truthy task
        task = Task(
            task_id="truthy_test",
            task_name="Truthy Test",
            task_type="truthy",
            level=0,
            prompt="Which response is better?",
            expected_answer={"type": "truthy", "conversation": []},
        )
        cl.session.add_task(task)

        # Mock llm_judge batch function
        with patch.object(
            cl.aux_reward_functions["llm_judge"], "compute_rewards_batch"
        ) as mock_batch:
            mock_result = MagicMock()
            mock_result.score = 0.75
            mock_result.info = "Good response"
            # Return list of lists (one per task, one per generation)
            mock_batch.return_value = [[mock_result]]

            # compute_reward returns combined score (with batch processing in num_generations=1 mode)
            primary_reward = cl.compute_reward("truthy_test", "Test response")

            # Score should be the judge score for truthy (aux_weight=0.0, so just judge score)
            self.assertAlmostEqual(
                primary_reward, 0.75, places=5
            )  # aux_weight=0.0, so just judge score

            # CRITICAL: Truthy tasks always have is_correct=False (never affects curriculum)
            task_from_session = cl.session.get_task("truthy_test")
            self.assertFalse(task_from_session.is_correct)

        # Create a new CurriculumLearning with format checking enabled for format gate test
        cl_format = CurriculumLearning(
            use_llm_judge=True,
            use_format=True,  # Enable format checking
            aux_weight=0.0,  # Disable auxiliary weight blending for clean test
            num_generations=1,  # Single generation mode for immediate processing
            llm_judge_kwargs={
                "api_host": "localhost",
                "api_port": 8000,
                "model_name": "Skywork/Skywork-Reward-V2-Qwen3-4B",
            },
            use_whitespace_collapse=False,  # Disable whitespace collapse
            use_lang_consistency=False,  # Disable auxiliary rewards to test just LLM Judge + format
            use_repetition=False,
            use_reasoning_steps=False,
            use_length=False,
        )

        # Set up mock tokenizer for LLM Judge
        setup_llm_judge_with_mock_tokenizer(cl_format)

        # Create a truthy task
        task = Task(
            task_id="truthy_format_test",
            task_name="Format Gate Test",
            task_type="truthy",
            level=0,
            prompt="Which response is better?",
            expected_answer={"type": "truthy", "conversation": []},
        )
        cl_format.session.add_task(task)

        # Mock llm_judge batch function to return high score
        with patch.object(
            cl_format.aux_reward_functions["llm_judge"], "compute_rewards_batch"
        ) as mock_batch:
            mock_result = MagicMock()
            mock_result.score = 0.9  # High judge score
            mock_result.info = "Excellent response"
            # Return list of lists (one per task, one per generation)
            mock_batch.return_value = [[mock_result]]

            # Response missing answer tag (format invalid)
            response_invalid_format = "No answer tag here"
            primary_reward = cl_format.compute_reward(
                "truthy_format_test", response_invalid_format
            )

            # Score should be the judge score (format gate removed)
            self.assertEqual(
                primary_reward,
                0.9,
                "Judge reward should NOT be gated when format is invalid (format gate removed)",
            )

    def test_batch_llm_judge_validates_request_payload(self):
        """Test that batch LLM Judge request payload contains correct number of tasks."""
        cl = CurriculumLearning(
            use_llm_judge=True,
            aux_weight=0.0,  # Disable auxiliary weight blending for clean test
            num_generations=1,  # Single generation mode
            llm_judge_kwargs={
                "api_host": "localhost",
                "api_port": 8000,
                "model_name": "Skywork/Skywork-Reward-V2-Qwen3-4B",
            },
            use_format=False,  # Disable format to avoid auxiliary reward blending
            use_whitespace_collapse=False,  # Disable whitespace collapse
            use_lang_consistency=False,  # Disable auxiliary rewards for clean test
            use_repetition=False,
        )

        # Set up mock tokenizer for LLM Judge
        setup_llm_judge_with_mock_tokenizer(cl)

        # Create multiple truthy tasks
        task_ids = []
        for i in range(3):
            task = Task(
                task_id=f"truthy_{i}",
                task_name=f"Truthy {i}",
                task_type="truthy",
                level=0,
                prompt=f"Which response is better (task {i})?",
                expected_answer={"type": "truthy", "conversation": []},
            )
            cl.session.add_task(task)
            task.model_output = f"Response {i}"
            task_ids.append(f"truthy_{i}")

        # Mock the batch judge function to return results
        with patch.object(
            cl.aux_reward_functions["llm_judge"], "compute_rewards_batch"
        ) as mock_batch:
            # Create mock results for each separate batch call
            # With num_generations=1, each task gets batch processed independently
            mock_batch.side_effect = [
                [[MagicMock(score=0.0, info="Response 0 quality")]],  # First task
                [[MagicMock(score=0.5, info="Response 1 quality")]],  # Second task
                [[MagicMock(score=1.0, info="Response 2 quality")]],  # Third task
            ]

            # compute_reward for each task returns combined score directly
            rewards = []
            for task_id in task_ids:
                score = cl.compute_reward(task_id, "Test")
                rewards.append(score)

            # Verify batch API was called 3 times (once per task in single-generation mode)
            self.assertEqual(mock_batch.call_count, 3)

            # Verify final rewards use judge scores (no normalization since aux_weight=0.0)
            self.assertAlmostEqual(rewards[0], 0.0, places=5)  # First task judge score
            self.assertAlmostEqual(rewards[1], 0.5, places=5)  # Second task judge score
            self.assertAlmostEqual(rewards[2], 1.0, places=5)  # Third task judge score

    def test_batch_llm_judge_with_mixed_task_types(self):
        """Test LLM Judge handles both truthy and math/puzzle tasks correctly."""
        cl = CurriculumLearning(
            use_llm_judge=True,
            aux_weight=0.0,  # Disable auxiliary weight blending for clean test
            llm_judge_weight=0.0,  # Disable judge weight for clean test
            num_generations=1,  # Single generation mode
            llm_judge_kwargs={
                "api_host": "localhost",
                "api_port": 8000,
                "model_name": "Skywork/Skywork-Reward-V2-Qwen3-4B",
            },
            use_format=False,  # Disable auxiliary rewards for clean test
            use_whitespace_collapse=False,
            use_lang_consistency=False,
            use_repetition=False,
        )

        # Set up mock tokenizer for LLM Judge
        setup_llm_judge_with_mock_tokenizer(cl)

        # Create 1 truthy and 1 math task
        truthy_task = Task(
            task_id="truthy_0",
            task_name="Truthy",
            task_type="truthy",
            level=0,
            prompt="Which is better?",
            expected_answer={"type": "truthy", "conversation": []},
        )
        truthy_task.model_output = "Response"
        cl.session.add_task(truthy_task)

        math_task = Task(
            task_id="math_0",
            task_name="Math",
            task_type="math",
            level=0,
            prompt="What is 2+2?",
            expected_answer="4",
        )
        math_task.model_output = "<think>2+2 equals 4</think><answer>4</answer>"
        cl.session.add_task(math_task)

        # Mock LLM Judge batch function
        with patch.object(
            cl.aux_reward_functions["llm_judge"], "compute_rewards_batch"
        ) as mock_batch:
            # With num_generations=1, each task gets batch processed individually
            # First batch call is for truthy_0, second is for math_0
            mock_batch.side_effect = [
                [[MagicMock(score=0.8, info="Quality 0")]],  # For truthy
                [[MagicMock(score=0.8, info="Quality 1")]],  # For math
            ]

            # Mock math reward function
            with patch.object(
                cl.reward_functions["math"], "compute_reward"
            ) as mock_math_reward:
                mock_math_result = MagicMock()
                mock_math_result.score = 1.0
                mock_math_result.info = "Correct"
                mock_math_reward.return_value = mock_math_result

                # Compute rewards for both
                truthy_reward = cl.compute_reward("truthy_0", "Response")
                math_reward = cl.compute_reward(
                    "math_0", "<think>2+2=4</think><answer>4</answer>"
                )

                # Truthy gets judge score as primary (aux_weight=0.0)
                self.assertAlmostEqual(truthy_reward, 0.8, places=5)

                # Math gets combined: primary (math correct=1.0) + aux (judge=0.8)
                # With aux_weight=0.0: combined = 1.0 * (1-0) + 0.8 * 0 = 1.0
                self.assertAlmostEqual(math_reward, 1.0, places=5)

    def test_batch_llm_judge_with_fallback_to_individual(self):
        """Test LLM Judge with individual compute_reward calls."""
        cl = CurriculumLearning(
            use_llm_judge=True,
            aux_weight=0.0,  # Disable auxiliary weight blending for clean test
            num_generations=1,  # Single generation mode
            llm_judge_kwargs={
                "api_host": "localhost",
                "api_port": 8000,
                "model_name": "Skywork/Skywork-Reward-V2-Qwen3-4B",
            },
            use_format=False,  # Disable auxiliary to avoid extra calls
            use_whitespace_collapse=False,
            use_lang_consistency=False,
            use_repetition=False,
        )

        # Set up mock tokenizer for LLM Judge
        setup_llm_judge_with_mock_tokenizer(cl)

        # Create multiple tasks
        task_ids = []
        for i in range(2):
            task = Task(
                task_id=f"truthy_{i}",
                task_name=f"Truthy {i}",
                task_type="truthy",
                level=0,
                prompt="Which is better?",
                expected_answer={"type": "truthy", "conversation": []},
            )
            cl.session.add_task(task)
            task.model_output = f"Response {i}"
            task_ids.append(f"truthy_{i}")

        # Mock the batch judge function
        with patch.object(
            cl.aux_reward_functions["llm_judge"], "compute_rewards_batch"
        ) as mock_batch:
            # With num_generations=1, each task gets batch processed independently
            mock_batch.side_effect = [
                [[MagicMock(score=0.0, info="Quality 0")]],  # First task
                [[MagicMock(score=1.0, info="Quality 1")]],  # Second task
            ]

            # Compute rewards for each task
            rewards = []
            for task_id in task_ids:
                score = cl.compute_reward(task_id, "Test")
                rewards.append(score)

            # Verify results (no normalization since aux_weight=0.0)
            self.assertAlmostEqual(rewards[0], 0.0, places=5)
            self.assertAlmostEqual(rewards[1], 1.0, places=5)

    def test_truthy_task_does_not_affect_curriculum(self):
        """Test that truthy tasks never affect curriculum success windows."""
        cl = CurriculumLearning(
            use_llm_judge=True,
            llm_judge_kwargs={
                "api_host": "localhost",
                "api_port": 8000,
                "model_name": "Skywork",
            },
            num_generations=1,  # Single evaluation mode for immediate tracking
        )

        cl.current_level = 0

        # Create and evaluate 20 truthy tasks with high judge scores
        for i in range(20):
            task = Task(
                task_id=f"truthy_{i}",
                task_name=f"Truthy {i}",
                task_type="truthy",
                level=0,
                prompt="Which response is better?",
                expected_answer={"type": "truthy", "conversation": []},
            )
            cl.session.add_task(task)

            # Mock high LLM Judge scores
            with patch.object(
                cl.aux_reward_functions["llm_judge"], "compute_reward"
            ) as mock_judge:
                mock_result = MagicMock()
                mock_result.score = 0.95  # Very high quality
                mock_result.info = "Excellent"
                mock_judge.return_value = mock_result

                cl.compute_reward(f"truthy_{i}", "Response")

        # Trigger batch processing to ensure judge scores are computed
        judge_stats = cl.get_judge_scores()

        # Check that level did NOT advance despite high truthy scores
        # Truthy tasks should NOT contribute to success windows
        self.assertEqual(
            cl.current_level,
            0,
            "Truthy tasks should not trigger curriculum advancement",
        )

    # === NEW TESTS FOR REFACTORED METHODS ===

    def test_compute_reward_truthy_returns_placeholder_score(self):
        """Test _compute_reward_truthy uses placeholder score (batch LLM Judge via get_rewards)."""
        cl = CurriculumLearning(
            use_llm_judge=True,
            llm_judge_kwargs={
                "api_host": "localhost",
                "api_port": 8000,
                "model_name": "Skywork",
            },
        )

        task = Task(
            task_id="truthy_test",
            task_name="Truthy Test",
            task_type="truthy",
            level=0,
            prompt="Which is better?",
            expected_answer={"type": "truthy", "conversation": []},
        )
        cl.session.add_task(task)
        task.model_output = "My response"

        # Call the refactored method directly (no judge computation)
        score, is_correct, task_rewards, aux_score_dict = cl._compute_reward_truthy(
            task
        )

        # Verify returned values: placeholder 0.0 (LLM Judge deferred to batch)
        self.assertEqual(
            score, 0.0, "Score should be placeholder (LLM Judge deferred to batch)"
        )
        self.assertFalse(is_correct, "Truthy always has is_correct=False")
        self.assertEqual(
            task_rewards[0].score,
            0.0,
            "Primary reward is placeholder pending batch",
        )
        self.assertEqual(task_rewards[0].reward_function_name, "primary")

    def test_compute_reward_standard_success_path(self):
        """Test _compute_reward_standard with format valid and correct answer."""
        cl = CurriculumLearning(use_format=True)

        task = Task(
            task_id="math_test",
            task_name="Math Test",
            task_type="math",
            level=1,
            prompt="What is 2+2?",
            expected_answer="4",
        )
        cl.session.add_task(task)
        task.model_output = "<think>2+2=4</think>\n<answer>4</answer>"

        # Mock the task reward function (returns 1.0 for correct)
        with patch.object(
            cl.reward_functions["math"], "compute_reward"
        ) as mock_task_fn:
            mock_result = MagicMock()
            mock_result.score = 1.0
            mock_result.info = "Correct"
            mock_task_fn.return_value = mock_result

            # Call the refactored method
            score, is_correct, task_rewards, aux_score_dict = (
                cl._compute_reward_standard(task)
            )

            # Verify success path
            self.assertEqual(score, 1.0, "Score should be 1.0 on success")
            self.assertTrue(is_correct, "is_correct should be True")
            self.assertEqual(
                task_rewards[0].score, 1.0, "Primary reward is task correctness"
            )

    def test_compute_reward_standard_failure_format_invalid(self):
        """Test _compute_reward_standard with invalid format."""
        cl = CurriculumLearning(use_format=True)

        task = Task(
            task_id="math_test",
            task_name="Math Test",
            task_type="math",
            level=1,
            prompt="What is 2+2?",
            expected_answer="4",
        )
        cl.session.add_task(task)
        task.model_output = "<answer>4</answer>"  # Missing <think> tag

        # Mock the task reward function (returns 1.0 even though format invalid)
        with patch.object(
            cl.reward_functions["math"], "compute_reward"
        ) as mock_task_fn:
            mock_result = MagicMock()
            mock_result.score = 1.0
            mock_result.info = "Correct"
            mock_task_fn.return_value = mock_result

            # Call the refactored method
            score, is_correct, task_rewards, aux_score_dict = (
                cl._compute_reward_standard(task)
            )

            # Verify failure path
            self.assertEqual(score, 0.0, "Score should be 0.0 on format failure")
            self.assertFalse(is_correct, "is_correct should be False")

    def test_compute_reward_standard_failure_incorrect_answer(self):
        """Test _compute_reward_standard with incorrect answer."""
        cl = CurriculumLearning(use_format=True)

        task = Task(
            task_id="math_test",
            task_name="Math Test",
            task_type="math",
            level=1,
            prompt="What is 2+2?",
            expected_answer="4",
        )
        cl.session.add_task(task)
        task.model_output = "<think>2+2=5</think>\n<answer>5</answer>"

        # Mock the task reward function (returns 0.0 for incorrect)
        with patch.object(
            cl.reward_functions["math"], "compute_reward"
        ) as mock_task_fn:
            mock_result = MagicMock()
            mock_result.score = 0.0
            mock_result.info = "Incorrect"
            mock_task_fn.return_value = mock_result

            # Call the refactored method
            score, is_correct, task_rewards, aux_score_dict = (
                cl._compute_reward_standard(task)
            )

            # Verify failure path
            self.assertEqual(score, 0.0, "Score should be 0.0 on incorrect answer")
            self.assertFalse(is_correct, "is_correct should be False")

    def test_check_format_validity_both_tags_valid(self):
        """Test _check_format_validity returns True when both tags valid."""
        from infinite_rl.generation import Generation

        cl = CurriculumLearning(use_format=True)

        task = Task(
            task_id="test",
            task_name="Test",
            task_type="math",
            level=0,
            prompt="Test",
            expected_answer="4",
        )
        cl.session.add_task(task)

        # Create a generation with valid format
        output = "<think>Reasoning here</think>\n<answer>4</answer>"
        gen = Generation(output=output, rewards=[], primary_score=1.0)

        # Mock both format functions to return 1.0 (valid)
        with patch.object(
            cl.aux_reward_functions["format_think"], "compute_reward"
        ) as mock_think:
            mock_result = MagicMock()
            mock_result.score = 1.0
            mock_result.info = "Valid think tag"
            mock_think.return_value = mock_result

            with patch.object(
                cl.aux_reward_functions["format_answer"], "compute_reward"
            ) as mock_answer:
                mock_result = MagicMock()
                mock_result.score = 1.0
                mock_result.info = "Valid answer tag"
                mock_answer.return_value = mock_result

                valid, reason = cl._check_format_validity(task, gen)

                self.assertTrue(valid, "Format should be valid")
                self.assertEqual(reason, "", "No failure reason when valid")

    def test_check_format_validity_think_tag_invalid(self):
        """Test _check_format_validity returns False when think tag invalid."""
        from infinite_rl.generation import Generation

        cl = CurriculumLearning(use_format=True)

        task = Task(
            task_id="test",
            task_name="Test",
            task_type="math",
            level=0,
            prompt="Test",
            expected_answer="4",
        )
        cl.session.add_task(task)

        # Create a generation with missing think tag
        output = "<answer>4</answer>"
        gen = Generation(output=output, rewards=[], primary_score=1.0)

        # Mock format_think to return < 1.0 (invalid)
        with patch.object(
            cl.aux_reward_functions["format_think"], "compute_reward"
        ) as mock_think:
            mock_result = MagicMock()
            mock_result.score = 0.0
            mock_result.info = "Missing think tag"
            mock_think.return_value = mock_result

            with patch.object(
                cl.aux_reward_functions["format_answer"], "compute_reward"
            ) as mock_answer:
                mock_result = MagicMock()
                mock_result.score = 1.0
                mock_result.info = "Valid answer tag"
                mock_answer.return_value = mock_result

                valid, reason = cl._check_format_validity(task, gen)

                self.assertFalse(valid, "Format should be invalid")
                self.assertIn("think", reason, "Reason should mention think tag")

    def test_finalize_reward_batch_single_generation(self):
        """Test compute_reward with single generation (no GRPO batching)."""
        cl = CurriculumLearning(num_generations=1, aux_weight=0.0, use_format=False)
        cl.current_level = 0

        task = Task(
            task_id="math_test",
            task_name="Math Test",
            task_type="math",
            level=0,
            prompt="Test",
            expected_answer="4",
        )
        cl.session.add_task(task)

        # Mock the reward function
        with patch.object(
            cl.reward_functions["math"], "compute_reward"
        ) as mock_compute:
            mock_result = MagicMock()
            mock_result.score = 1.0
            mock_result.info = "Correct"
            mock_compute.return_value = mock_result

            # Compute reward returns combined score directly for single generation
            score = cl.compute_reward("math_test", "4")

        # Should return the combined reward
        self.assertEqual(score, 1.0)
        # Check that success was tracked (via step counter)
        self.assertEqual(cl.global_step, 1, "Should increment global_step after batch")

    def test_finalize_reward_batch_grpo_accumulation(self):
        """Test compute_reward accumulates GRPO responses until batch complete."""
        cl = CurriculumLearning(
            num_generations=3,
            aux_weight=0.0,  # Disable auxiliary blending for clean test
            use_format=False,  # Disable format checking
            use_repetition=False,
            use_lang_consistency=False,
            use_whitespace_collapse=False,
            use_length=False,
            use_reasoning_steps=False,
        )
        cl.current_level = 0

        # Create task
        base_task_id = "prompt_base"
        task = Task(
            task_id=base_task_id,
            task_name="Base Task",
            task_type="math",
            level=0,
            prompt="What is 2+2?",
            expected_answer="4",
        )
        cl.session.add_task(task)

        # Mock the reward function
        with patch.object(
            cl.reward_functions["math"], "compute_reward"
        ) as mock_compute:
            mock_result = MagicMock()
            mock_result.score = 1.0
            mock_result.info = "Correct"
            mock_compute.return_value = mock_result

            # Simulate 3 generations for the same task
            rewards = []
            for i in range(3):
                score = cl.compute_reward(base_task_id, f"Response {i}")
                rewards.append(score)

        # Should return combined reward for the batch
        self.assertEqual(len(rewards), 3)
        # For incomplete batches (first 2 calls), return primary score as estimate
        # For complete batch (last call), return final combined score
        self.assertEqual(rewards[0], 1.0)  # Primary score for generation 0
        self.assertEqual(rewards[1], 1.0)  # Primary score for generation 1
        self.assertEqual(rewards[2], 1.0)  # Final combined score when batch complete

        # After batch complete, global_step should be 1 (one batch processed)
        self.assertEqual(cl.global_step, 1, "Batch complete should increment step once")

        # Task should have accumulated all 3 generations
        self.assertEqual(len(task.generations), 3, "Task should have 3 generations")
        self.assertEqual(task.generations[0].output, "Response 0")
        self.assertEqual(task.generations[1].output, "Response 1")
        self.assertEqual(task.generations[2].output, "Response 2")

    def test_finalize_reward_batch_excludes_truthy_from_curriculum(self):
        """Test compute_reward excludes truthy tasks from success tracking."""
        cl = CurriculumLearning(
            use_llm_judge=True,
            llm_judge_kwargs={
                "api_host": "localhost",
                "api_port": 8000,
                "model_name": "Skywork",
            },
            num_generations=1,
        )
        cl.current_level = 0

        task = Task(
            task_id="truthy_test",
            task_name="Truthy Test",
            task_type="truthy",
            level=0,
            prompt="Which is better?",
            expected_answer={"type": "truthy", "conversation": []},
        )
        cl.session.add_task(task)

        # Set up mock tokenizer for LLM Judge
        setup_llm_judge_with_mock_tokenizer(cl)

        # Mock LLM Judge for truthy
        with patch.object(
            cl.aux_reward_functions["llm_judge"], "compute_reward"
        ) as mock_judge:
            mock_result = MagicMock()
            mock_result.score = 0.9
            mock_result.info = "Good"
            mock_judge.return_value = mock_result

            # Compute reward (for truthy, this sets up the task)
            score = cl.compute_reward("truthy_test", "Some response")

        # Truthy should NOT contribute to success windows
        # Level should remain at 0 (no advancement from truthy alone)
        self.assertEqual(
            cl.current_level, 0, "Truthy tasks should not advance curriculum"
        )

    def test_compute_reward_dispatches_to_correct_handler(self):
        """Test that compute_reward dispatcher calls appropriate handler for task type."""
        cl = CurriculumLearning(
            use_format=True,
            use_llm_judge=True,
            llm_judge_kwargs={
                "api_host": "localhost",
                "api_port": 8000,
                "model_name": "Skywork",
            },
        )

        # Test math task dispatch
        math_task = Task(
            task_id="math_test",
            task_name="Math",
            task_type="math",
            level=0,
            prompt="Test",
            expected_answer="4",
        )
        cl.session.add_task(math_task)

        with patch.object(cl, "_compute_reward_standard") as mock_standard:
            mock_standard.return_value = (1.0, True, [], {})
            with patch.object(
                cl.reward_functions["math"], "compute_reward"
            ) as mock_task_fn:
                mock_result = MagicMock()
                mock_result.score = 1.0
                mock_result.info = ""
                mock_task_fn.return_value = mock_result

                cl.compute_reward(
                    "math_test", "<think>Test</think>\n<answer>4</answer>"
                )

                # Should call _compute_reward_standard for math task
                mock_standard.assert_called_once()

        # Test truthy task dispatch
        truthy_task = Task(
            task_id="truthy_test",
            task_name="Truthy",
            task_type="truthy",
            level=0,
            prompt="Test",
            expected_answer={},
        )
        cl.session.add_task(truthy_task)

        with patch.object(cl, "_compute_reward_truthy") as mock_truthy:
            mock_truthy.return_value = (0.8, False, [], {})
            with patch.object(
                cl.aux_reward_functions["llm_judge"], "compute_reward"
            ) as mock_judge:
                mock_result = MagicMock()
                mock_result.score = 0.8
                mock_result.info = ""
                mock_judge.return_value = mock_result

                cl.compute_reward("truthy_test", "Response")

                # Should call _compute_reward_truthy for truthy task
                mock_truthy.assert_called_once()


if __name__ == "__main__":
    unittest.main()
