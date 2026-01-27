import unittest
from infinite_rl.puzzles import PuzzlePrompts
from unittest.mock import patch


class TestPuzzlePrompts(unittest.TestCase):
    """Test PuzzlePrompts class and related functions."""

    def setUp(self):
        """Set up test fixtures."""
        self.test_data = {
            "javascript": {
                "TestPuzzle": {
                    "name": "TestPuzzle",
                    "language": "javascript",
                    "prompt": "Test prompt for JavaScript puzzle",
                },
                "AnotherPuzzle": {
                    "name": "AnotherPuzzle",
                    "language": "javascript",
                    "prompt": "Another test prompt",
                },
            },
            "python": {
                "PythonPuzzle": {
                    "name": "PythonPuzzle",
                    "language": "python",
                    "prompt": "Test prompt for Python puzzle",
                }
            },
        }

    def test_get_puzzle_prompt_existing_javascript(self):
        """Test getting prompt for existing JavaScript puzzle."""
        prompts = PuzzlePrompts()

        # Manually load test data
        prompts._prompts = {}
        for language, puzzles in self.test_data.items():
            for puzzle_name, data in puzzles.items():
                prompts._prompts[f"{language}/{puzzle_name}"] = data
        prompts._loaded = True

        result = prompts.get_puzzle_data("TestPuzzle", "javascript")
        self.assertIsNotNone(result)

    def test_get_puzzle_prompt_existing_python(self):
        """Test getting prompt for existing Python puzzle."""
        prompts = PuzzlePrompts()

        # Manually load test data
        prompts._prompts = {}
        for language, puzzles in self.test_data.items():
            for puzzle_name, data in puzzles.items():
                prompts._prompts[f"{language}/{puzzle_name}"] = data
        prompts._loaded = True

        result = prompts.get_puzzle_data("PythonPuzzle", "python")
        self.assertIsNotNone(result)

    def test_get_puzzle_prompt_non_existing(self):
        """Test getting prompt for non-existing puzzle."""
        prompts = PuzzlePrompts()

        # Manually load test data
        prompts._prompts = {}
        for language, puzzles in self.test_data.items():
            for puzzle_name, data in puzzles.items():
                prompts._prompts[f"{language}/{puzzle_name}"] = data
        prompts._loaded = True

        result = prompts.get_puzzle_data("NonExistingPuzzle", "javascript")
        self.assertIsNone(result)

    def test_get_puzzle_prompt_wrong_language(self):
        """Test getting prompt for puzzle in wrong language."""
        prompts = PuzzlePrompts()

        # Manually load test data
        prompts._prompts = {}
        for language, puzzles in self.test_data.items():
            for puzzle_name, data in puzzles.items():
                prompts._prompts[f"{language}/{puzzle_name}"] = data
        prompts._loaded = True

        result = prompts.get_puzzle_data("TestPuzzle", "python")
        self.assertIsNone(result)

    def test_get_available_puzzles_javascript(self):
        """Test getting available puzzles for JavaScript."""
        prompts = PuzzlePrompts()

        # Manually load test data
        prompts._prompts = {}
        for language, puzzles in self.test_data.items():
            for puzzle_name, data in puzzles.items():
                prompts._prompts[f"{language}/{puzzle_name}"] = data
        prompts._loaded = True

        result = prompts.get_available_puzzles("javascript")
        self.assertEqual(set(result), {"TestPuzzle", "AnotherPuzzle"})

    def test_get_available_puzzles_python(self):
        """Test getting available puzzles for Python."""
        prompts = PuzzlePrompts()

        # Manually load test data
        prompts._prompts = {}
        for language, puzzles in self.test_data.items():
            for puzzle_name, data in puzzles.items():
                prompts._prompts[f"{language}/{puzzle_name}"] = data
        prompts._loaded = True

        result = prompts.get_available_puzzles("python")
        self.assertEqual(result, ["PythonPuzzle"])

    @patch.object(PuzzlePrompts, "_load_prompts")
    def test_unloaded_prompts(self, mock_load):
        """Test behavior when prompts are not loaded."""
        # Mock _load_prompts to do nothing, so prompts stay unloaded
        mock_load.return_value = None

        prompts = PuzzlePrompts()
        # Don't load anything
        result = prompts.get_puzzle_data("TestPuzzle", "javascript")
        self.assertIsNone(result)

        result = prompts.get_available_puzzles("javascript")
        self.assertEqual(result, [])


if __name__ == "__main__":
    unittest.main()
