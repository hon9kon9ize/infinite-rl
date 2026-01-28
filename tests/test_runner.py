import unittest
from unittest.mock import patch, MagicMock
import json
import sys
from infinite_rl import runner


class TestRunner(unittest.TestCase):
    """Test runner module functions."""

    def setUp(self):
        """Set up test fixtures."""
        # Ensure puzzles dict is available and mock it
        runner.puzzles = {}

    @patch("infinite_rl.runner.executor")
    def test_eval_puzzle_javascript_success(self, mock_executor):
        """Test evaluating JavaScript puzzle successfully."""
        # Mock puzzle in puzzles dict
        mock_puzzle_class = MagicMock()
        mock_puzzle_class.sat.return_value = True  # Mock the sat method to return True
        runner.puzzles["TestPuzzle"] = mock_puzzle_class

        mock_executor.run_single.return_value = (
            '{"result": 42, "isCorrect": true}',
            "",
        )

        result = runner.evalPuzzle(
            "TestPuzzle", "function sol() { return 42; }", {"input": 1}, "javascript"
        )

        self.assertEqual(result, {"result": 42, "isCorrect": True})
        mock_executor.run_single.assert_called_once()

    @patch("infinite_rl.runner.executor")
    def test_eval_puzzle_javascript_error(self, mock_executor):
        """Test evaluating JavaScript puzzle with execution error."""
        mock_executor.run_single.return_value = (None, "Syntax error")

        result = runner.evalPuzzle(
            "TestPuzzle", "invalid code", {"input": 1}, "javascript"
        )

        self.assertEqual(result, {"error": "Syntax error"})

    @patch("infinite_rl.runner.executor")
    def test_eval_puzzle_javascript_invalid_json(self, mock_executor):
        """Test evaluating JavaScript puzzle with invalid JSON output."""
        mock_executor.run_single.return_value = ("invalid json", "")

        result = runner.evalPuzzle("TestPuzzle", "code", {"input": 1}, "javascript")

        self.assertIn("Invalid JSON output", result["error"])

    @patch("infinite_rl.runner.executor")
    def test_eval_puzzle_javascript_js_error(self, mock_executor):
        """Test evaluating JavaScript puzzle with JS-side error."""
        mock_executor.run_single.return_value = ('{"error": "JS error"}', "")

        result = runner.evalPuzzle("TestPuzzle", "code", {"input": 1}, "javascript")

        self.assertEqual(result, {"error": "JS error"})

    @patch("builtins.exec")
    def test_eval_puzzle_python_success(self, mock_exec):
        """Test evaluating Python puzzle successfully."""
        # Mock the puzzle class
        mock_puzzle_class = MagicMock()
        mock_puzzle_class.sat.return_value = True
        runner.puzzles = {"TestPuzzle": mock_puzzle_class}

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
    def test_eval_puzzle_python_incorrect(self, mock_exec):
        """Test evaluating Python puzzle with incorrect solution."""
        # Mock the puzzle class
        mock_puzzle_class = MagicMock()
        mock_puzzle_class.sat.return_value = False
        runner.puzzles = {"TestPuzzle": mock_puzzle_class}

        # Mock exec to define sol function
        def mock_exec_func(code, globals_dict):
            globals_dict["sol"] = lambda x: 0

        mock_exec.side_effect = mock_exec_func

        result = runner.evalPuzzle(
            "TestPuzzle", "def sol(x): return 0", {"input": 1}, "python"
        )

        self.assertEqual(result, {"result": 0, "isCorrect": False})

    def test_eval_puzzle_unknown_puzzle(self):
        """Test evaluating puzzle with unknown puzzle name."""
        runner.puzzles = {}  # Clear puzzles

        result = runner.evalPuzzle("UnknownPuzzle", "code", {"input": 1}, "python")

        self.assertEqual(result, {"error": "Unknown puzzle: UnknownPuzzle"})

    @patch("builtins.exec")
    def test_eval_puzzle_python_exception(self, mock_exec):
        """Test evaluating Python puzzle with execution exception."""
        # Mock puzzle in puzzles dict so it gets past the unknown puzzle check
        mock_puzzle_class = MagicMock()
        runner.puzzles = {"TestPuzzle": mock_puzzle_class}

        # Mock exec to raise an exception
        mock_exec.side_effect = Exception("Syntax error")

        result = runner.evalPuzzle("TestPuzzle", "invalid code", {"input": 1}, "python")

        self.assertIn("Syntax error", result["error"])
        self.assertIn("stack", result)

    @patch("builtins.exec")
    def test_eval_puzzle_python_no_inputs(self, mock_exec):
        """Test evaluating Python puzzle with no inputs."""
        # Mock the puzzle class
        mock_puzzle_class = MagicMock()
        mock_puzzle_class.sat.return_value = True
        runner.puzzles = {"TestPuzzle": mock_puzzle_class}

        # Mock exec to define sol function
        def mock_exec_func(code, globals_dict):
            globals_dict["sol"] = lambda: 42

        mock_exec.side_effect = mock_exec_func

        result = runner.evalPuzzle("TestPuzzle", "def sol(): return 42", {}, "python")

        self.assertEqual(result, {"result": 42, "isCorrect": True})
        mock_puzzle_class.sat.assert_called_once_with(42)

    def test_eval_puzzle_case_insensitive_language(self):
        """Test that language parameter is case insensitive."""
        with patch("infinite_rl.runner.executor") as mock_executor:
            # Mock puzzle in puzzles dict
            mock_puzzle_class = MagicMock()
            mock_puzzle_class.sat.return_value = (
                True  # Mock the sat method to return True
            )
            runner.puzzles["TestPuzzle"] = mock_puzzle_class

            mock_executor.run_single.return_value = (
                '{"result": 42, "isCorrect": true}',
                "",
            )

            result = runner.evalPuzzle("TestPuzzle", "code", {"input": 1}, "JavaScript")

            mock_executor.run_single.assert_called_once()
            self.assertEqual(result, {"result": 42, "isCorrect": True})
