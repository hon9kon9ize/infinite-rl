"""
Test that JavaScript puzzles have valid example inputs.

This test ensures the bug where JavaScript puzzles had empty example
inputs ({}) is caught in CI.
"""

import unittest
import json
import os


class TestJavaScriptPuzzleExamples(unittest.TestCase):
    """Test that all JavaScript puzzles have proper example inputs."""

    def test_all_javascript_puzzles_have_examples(self):
        """Verify JavaScript puzzles have valid example inputs.

        Note: Empty dict {} is valid for puzzles where sol() takes no parameters.
        """
        # Find the puzzles.json file in the installed package
        import infinite_rl

        package_dir = os.path.dirname(infinite_rl.__file__)
        puzzles_json_path = os.path.join(package_dir, "runtimes", "puzzles.json")

        # Fallback to assets if runtimes doesn't exist (development mode)
        if not os.path.exists(puzzles_json_path):
            repo_root = os.path.dirname(os.path.dirname(__file__))
            puzzles_json_path = os.path.join(repo_root, "assets", "puzzles.json")

        self.assertTrue(
            os.path.exists(puzzles_json_path),
            f"puzzles.json not found at {puzzles_json_path}",
        )

        with open(puzzles_json_path, "r") as f:
            puzzles_data = json.load(f)

        javascript_puzzles = puzzles_data.get("javascript", {})
        self.assertGreater(
            len(javascript_puzzles), 0, "No JavaScript puzzles found in puzzles.json"
        )

        # Track puzzles with invalid examples (None or wrong type)
        # Empty dict {} is VALID for puzzles where sol() has no parameters
        puzzles_with_invalid_examples = []

        for puzzle_name, puzzle_info in javascript_puzzles.items():
            example = puzzle_info.get("example")

            # Check that example exists and is the correct type
            # Empty dict {} is valid!
            if example is None:
                puzzles_with_invalid_examples.append(
                    f"{puzzle_name}: missing 'example' field"
                )
            elif not isinstance(example, dict):
                puzzles_with_invalid_examples.append(
                    f"{puzzle_name}: 'example' is not a dict (type: {type(example).__name__})"
                )

        # Report failures - we expect ALL puzzles to have example field (even if {})
        if puzzles_with_invalid_examples:
            error_msg = (
                f"Found {len(puzzles_with_invalid_examples)} JavaScript puzzles "
                f"with invalid examples:\n"
                + "\n".join(f"  - {msg}" for msg in puzzles_with_invalid_examples[:10])
            )
            if len(puzzles_with_invalid_examples) > 10:
                error_msg += (
                    f"\n  ... and {len(puzzles_with_invalid_examples) - 10} more"
                )

            self.fail(error_msg)

    def test_specific_concat_strings_example(self):
        """Verify ConcatStrings has the expected example structure."""
        import infinite_rl

        package_dir = os.path.dirname(infinite_rl.__file__)
        puzzles_json_path = os.path.join(package_dir, "runtimes", "puzzles.json")

        if not os.path.exists(puzzles_json_path):
            repo_root = os.path.dirname(os.path.dirname(__file__))
            puzzles_json_path = os.path.join(repo_root, "assets", "puzzles.json")

        with open(puzzles_json_path, "r") as f:
            puzzles_data = json.load(f)

        concat_strings = puzzles_data.get("javascript", {}).get("ConcatStrings")
        self.assertIsNotNone(
            concat_strings, "ConcatStrings puzzle not found in JavaScript puzzles"
        )

        example = concat_strings.get("example")
        self.assertIsNotNone(example, "ConcatStrings has no 'example' field")
        self.assertIsInstance(example, dict, "ConcatStrings example is not a dict")
        self.assertGreater(len(example), 0, "ConcatStrings example is empty dict")

        # Verify expected parameters exist
        self.assertIn("s", example, "ConcatStrings example missing 's' parameter")
        self.assertIn("n", example, "ConcatStrings example missing 'n' parameter")

        # Verify parameter types
        self.assertIsInstance(
            example["s"],
            list,
            f"ConcatStrings example 's' should be list, got {type(example['s'])}",
        )
        self.assertIsInstance(
            example["n"],
            int,
            f"ConcatStrings example 'n' should be int, got {type(example['n'])}",
        )

    def test_javascript_examples_are_executable(self):
        """
        Test that JavaScript puzzle examples can be used for execution.

        This simulates what happens during actual puzzle evaluation.
        """
        from infinite_rl.utils.param_extractor import extract_puzzle_inputs
        import infinite_rl

        package_dir = os.path.dirname(infinite_rl.__file__)
        puzzles_json_path = os.path.join(package_dir, "runtimes", "puzzles.json")

        if not os.path.exists(puzzles_json_path):
            repo_root = os.path.dirname(os.path.dirname(__file__))
            puzzles_json_path = os.path.join(repo_root, "assets", "puzzles.json")

        with open(puzzles_json_path, "r") as f:
            puzzles_data = json.load(f)

        javascript_puzzles = puzzles_data.get("javascript", {})

        # Test a sample of JavaScript puzzles
        sample_puzzles = list(javascript_puzzles.items())[:5]

        for puzzle_name, puzzle_info in sample_puzzles:
            with self.subTest(puzzle=puzzle_name):
                # This should not raise an error
                inputs = extract_puzzle_inputs(puzzle_info, "javascript")

                self.assertIsNotNone(
                    inputs, f"{puzzle_name}: extract_puzzle_inputs returned None"
                )
                self.assertIsInstance(
                    inputs,
                    dict,
                    f"{puzzle_name}: inputs should be dict, got {type(inputs)}",
                )
                self.assertGreater(
                    len(inputs), 0, f"{puzzle_name}: inputs dict is empty"
                )


if __name__ == "__main__":
    unittest.main()
