import unittest
from unittest.mock import patch, MagicMock
import json
import sys
from infinite_rl import runner


class TestRunner(unittest.TestCase):
    """Test runner module functions."""

    def setUp(self):
        """Set up test fixtures."""
        # Clear puzzle loader cache before each test
        runner._puzzle_loader_cache.clear()

    @patch("infinite_rl.runner._get_executor")
    @patch("infinite_rl.runner._get_puzzle_class")
    def test_eval_puzzle_javascript_success(self, mock_get_puzzle, mock_get_executor):
        """Test evaluating JavaScript puzzle successfully."""
        # Mock puzzle class
        mock_puzzle_class = MagicMock()
        mock_puzzle_class.sat.return_value = True
        mock_get_puzzle.return_value = mock_puzzle_class

        # Mock executor
        mock_executor_instance = MagicMock()
        mock_executor_instance.run_single.return_value = (
            '{"result": 42}',
            "",
        )
        mock_get_executor.return_value = mock_executor_instance

        result = runner.evalPuzzle(
            "TestPuzzle", "function sol() { return 42; }", {"input": 1}, "javascript"
        )

        self.assertEqual(result, {"result": 42, "isCorrect": True})
        mock_executor_instance.run_single.assert_called_once()
        mock_puzzle_class.sat.assert_called_once_with(42, 1)

    @patch("infinite_rl.runner._get_executor")
    def test_eval_puzzle_javascript_error(self, mock_get_executor):
        """Test evaluating JavaScript puzzle with execution error."""
        mock_executor_instance = MagicMock()
        mock_executor_instance.run_single.return_value = (None, "Syntax error")
        mock_get_executor.return_value = mock_executor_instance

        result = runner.evalPuzzle(
            "TestPuzzle", "invalid code", {"input": 1}, "javascript"
        )

        self.assertEqual(result, {"error": "Syntax error"})

    @patch("infinite_rl.runner._get_executor")
    def test_eval_puzzle_javascript_invalid_json(self, mock_get_executor):
        """Test evaluating JavaScript puzzle with invalid JSON output."""
        mock_executor_instance = MagicMock()
        mock_executor_instance.run_single.return_value = ("invalid json", "")
        mock_get_executor.return_value = mock_executor_instance

        result = runner.evalPuzzle("TestPuzzle", "code", {"input": 1}, "javascript")

        self.assertIn("Invalid JSON output", result["error"])

    @patch("infinite_rl.runner._get_executor")
    @patch("infinite_rl.runner._get_puzzle_class")
    def test_eval_puzzle_javascript_js_error(self, mock_get_puzzle, mock_get_executor):
        """Test evaluating JavaScript puzzle with JS-side error."""
        mock_executor_instance = MagicMock()
        mock_executor_instance.run_single.return_value = ('{"error": "JS error"}', "")
        mock_get_executor.return_value = mock_executor_instance

        result = runner.evalPuzzle("TestPuzzle", "code", {"input": 1}, "javascript")

        self.assertEqual(result, {"error": "JS error"})

    @patch("builtins.exec")
    @patch("infinite_rl.runner._get_puzzle_class")
    def test_eval_puzzle_python_success(self, mock_get_puzzle, mock_exec):
        """Test evaluating Python puzzle successfully."""
        # Mock the puzzle class
        mock_puzzle_class = MagicMock()
        mock_puzzle_class.sat.return_value = True
        mock_get_puzzle.return_value = mock_puzzle_class

        # Mock exec to define sol function
        def mock_exec_func(code, globals_dict):
            globals_dict["sol"] = lambda x: 42

        mock_exec.side_effect = mock_exec_func

        result = runner.evalPuzzle(
            "TestPuzzle", "def sol(x): return 42", {"input": 1}, "python"
        )

        self.assertEqual(result, {"result": 42, "isCorrect": True})
        mock_puzzle_class.sat.assert_called_once_with(42, 1)

    @patch("builtins.exec")
    @patch("infinite_rl.runner._get_puzzle_class")
    def test_eval_puzzle_python_incorrect(self, mock_get_puzzle, mock_exec):
        """Test evaluating Python puzzle with incorrect solution."""
        # Mock the puzzle class
        mock_puzzle_class = MagicMock()
        mock_puzzle_class.sat.return_value = False
        mock_get_puzzle.return_value = mock_puzzle_class

        # Mock exec to define sol function
        def mock_exec_func(code, globals_dict):
            globals_dict["sol"] = lambda x: 0

        mock_exec.side_effect = mock_exec_func

        result = runner.evalPuzzle(
            "TestPuzzle", "def sol(x): return 0", {"input": 1}, "python"
        )

        self.assertEqual(result, {"result": 0, "isCorrect": False})

    @patch("infinite_rl.runner._get_puzzle_class")
    def test_eval_puzzle_unknown_puzzle(self, mock_get_puzzle):
        """Test evaluating puzzle with unknown puzzle name."""
        mock_get_puzzle.return_value = None  # Simulate puzzle not found

        result = runner.evalPuzzle(
            "UnknownPuzzle", "def sol(x): return 42", {"input": 1}, "python"
        )

        self.assertEqual(result, {"error": "Unknown puzzle: UnknownPuzzle"})

    @patch("builtins.exec")
    @patch("infinite_rl.runner._get_puzzle_class")
    def test_eval_puzzle_python_exception(self, mock_get_puzzle, mock_exec):
        """Test evaluating Python puzzle with execution exception."""
        # Mock puzzle class
        mock_puzzle_class = MagicMock()
        mock_get_puzzle.return_value = mock_puzzle_class

        # Mock exec to raise an exception
        mock_exec.side_effect = Exception("Syntax error")

        result = runner.evalPuzzle("TestPuzzle", "invalid code", {"input": 1}, "python")

        self.assertIn("Syntax error", result["error"])
        self.assertIn("stack", result)

    @patch("builtins.exec")
    @patch("infinite_rl.runner._get_puzzle_class")
    def test_eval_puzzle_python_no_inputs(self, mock_get_puzzle, mock_exec):
        """Test evaluating Python puzzle with no inputs."""
        # Mock the puzzle class
        mock_puzzle_class = MagicMock()
        mock_puzzle_class.sat.return_value = True
        mock_get_puzzle.return_value = mock_puzzle_class

        # Mock exec to define sol function
        def mock_exec_func(code, globals_dict):
            globals_dict["sol"] = lambda: 42

        mock_exec.side_effect = mock_exec_func

        result = runner.evalPuzzle("TestPuzzle", "def sol(): return 42", {}, "python")

        self.assertEqual(result, {"result": 42, "isCorrect": True})
        mock_puzzle_class.sat.assert_called_once_with(42)

    @patch("infinite_rl.runner._get_executor")
    @patch("infinite_rl.runner._get_puzzle_class")
    def test_eval_puzzle_case_insensitive_language(
        self, mock_get_puzzle, mock_get_executor
    ):
        """Test that language parameter is case insensitive."""
        # Mock puzzle class
        mock_puzzle_class = MagicMock()
        mock_puzzle_class.sat.return_value = True
        mock_get_puzzle.return_value = mock_puzzle_class

        # Mock executor
        mock_executor_instance = MagicMock()
        mock_executor_instance.run_single.return_value = (
            '{"result": 42}',
            "",
        )
        mock_get_executor.return_value = mock_executor_instance

        result = runner.evalPuzzle("TestPuzzle", "code", {"input": 1}, "JavaScript")

        mock_executor_instance.run_single.assert_called_once()
        self.assertEqual(result, {"result": 42, "isCorrect": True})
