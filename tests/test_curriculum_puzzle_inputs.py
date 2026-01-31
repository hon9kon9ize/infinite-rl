"""
Integration tests for puzzle input extraction in curriculum learning.

These tests verify that curriculum.py correctly extracts puzzle inputs
from sat() function signatures using the param_extractor utility.

This test suite would have caught the bug where puzzle inputs were always
set to empty dict {}, causing puzzle solvers to receive no arguments.
"""

import unittest
import tempfile
import json
from pathlib import Path
from infinite_rl.curriculum import CurriculumLearning, Task
from infinite_rl.utils.param_extractor import extract_puzzle_inputs


class TestCurriculumPuzzleInputExtraction(unittest.TestCase):
    """Integration tests for puzzle input extraction in curriculum."""

    def test_get_prompt_puzzle_task_has_extracted_inputs(self):
        """Puzzle tasks must include extracted inputs, not empty dict."""
        cl = CurriculumLearning()

        # Clear all levels except 1 to ensure we get only our mock puzzle task
        # Note: Puzzles start at level 1; level 0 is reserved for math tasks
        for level in range(0, 7):
            if level != 1:
                cl.tasks_by_level[level] = []

        # Manually set up a puzzle task with parameters in sat() at level 1
        cl.tasks_by_level[1] = [
            {
                "type": "puzzle",
                "language": "javascript",
                "puzzle_name": "DistinctChars",
                "data": {
                    "name": "DistinctChars",
                    "docstring": "Find distinct characters",
                    "sat": 'function sat(ans, s = "The quick brown fox", n = 28)',
                    "sol": "function sol(s, n) { return []; }",
                    "ans_type": "array",
                    "rating": 1,
                },
                "rating": 1,
                "id": "puzzle_distinct_chars",
            }
        ]

        task = cl.get_prompt()

        # Verify task was created
        self.assertIsNotNone(task)
        self.assertIsInstance(task, Task)
        self.assertEqual(task.task_type, "puzzle")

        # ✓ CRITICAL: Verify inputs are NOT empty
        expected_answer = task.expected_answer
        self.assertNotEqual(
            expected_answer["inputs"],
            {},
            "Puzzle inputs must not be empty - this was the bug!",
        )

        # ✓ Verify correct parameters were extracted
        self.assertIn("s", expected_answer["inputs"])
        self.assertIn("n", expected_answer["inputs"])
        self.assertEqual(expected_answer["inputs"]["s"], "The quick brown fox")
        self.assertEqual(expected_answer["inputs"]["n"], 28)

    def test_python_puzzle_task_has_extracted_inputs(self):
        """Python puzzle tasks must also have extracted inputs."""
        cl = CurriculumLearning()

        # Clear all levels except 1 to ensure we get only our mock task
        for level in range(0, 7):
            if level != 1:
                cl.tasks_by_level[level] = []

        cl.tasks_by_level[1] = [
            {
                "type": "puzzle",
                "language": "python",
                "puzzle_name": "StrIndex",
                "data": {
                    "name": "StrIndex",
                    "docstring": "Find string at index",
                    "sat": 'def sat(result, s="foobar", index=2):',
                    "sol": "def sol(s, index): return s[index]",
                    "ans_type": "str",
                    "rating": 1,
                },
                "rating": 1,
                "id": "puzzle_str_index",
            }
        ]

        task = cl.get_prompt()

        expected_answer = task.expected_answer
        self.assertNotEqual(expected_answer["inputs"], {})
        self.assertEqual(expected_answer["inputs"]["s"], "foobar")
        self.assertEqual(expected_answer["inputs"]["index"], 2)

    def test_puzzle_with_no_parameters_has_empty_inputs(self):
        """Puzzle with no default parameters should have empty inputs dict."""
        cl = CurriculumLearning()

        # Clear all levels except 1 to ensure we get only our mock task
        for level in range(0, 7):
            if level != 1:
                cl.tasks_by_level[level] = []

        cl.tasks_by_level[1] = [
            {
                "type": "puzzle",
                "language": "javascript",
                "puzzle_name": "SimpleAdd",
                "data": {
                    "name": "SimpleAdd",
                    "docstring": "Simple addition",
                    "sat": "function sat(result, a, b) { return result === a + b; }",
                    "sol": "function sol(a, b) { return a + b; }",
                    "ans_type": "number",
                    "rating": 1,
                },
                "rating": 1,
                "id": "puzzle_simple_add",
            }
        ]

        task = cl.get_prompt()

        # With no default parameters, inputs should be empty
        # (first param 'result' is what sol() returns, not an input)
        expected_answer = task.expected_answer
        self.assertEqual(
            expected_answer["inputs"],
            {},
            "Puzzles with no default parameters should have empty inputs",
        )

    def test_puzzle_inputs_match_sat_signature(self):
        """Extracted inputs must match the sat() function signature."""
        test_cases = [
            {
                "name": "DistinctChars",
                "language": "javascript",
                "sat": 'function sat(ans, s = "hello", n = 5)',
                "expected": {"s": "hello", "n": 5},
            },
            {
                "name": "StrIndex",
                "language": "python",
                "sat": 'def sat(r, s="test", idx=0):',
                "expected": {"s": "test", "idx": 0},
            },
            {
                "name": "ArraySum",
                "language": "javascript",
                "sat": "function sat(result, arr = [1, 2, 3])",
                "expected": {"arr": [1, 2, 3]},
            },
        ]

        for tc in test_cases:
            with self.subTest(puzzle=tc["name"]):
                cl = CurriculumLearning()

                # Clear all levels except 1 to ensure we get only our mock task
                for level in range(0, 7):
                    if level != 1:
                        cl.tasks_by_level[level] = []

                cl.tasks_by_level[1] = [
                    {
                        "type": "puzzle",
                        "language": tc["language"],
                        "puzzle_name": tc["name"],
                        "data": {
                            "name": tc["name"],
                            "docstring": "Test",
                            "sat": tc["sat"],
                            "sol": "function sol() {}",
                            "ans_type": "any",
                            "rating": 1,
                        },
                        "rating": 1,
                        "id": f"puzzle_{tc['name']}",
                    }
                ]

                task = cl.get_prompt()
                inputs = task.expected_answer["inputs"]

                self.assertEqual(
                    inputs,
                    tc["expected"],
                    f"Inputs for {tc['name']} don't match sat() signature",
                )

    def test_puzzle_complex_default_values(self):
        """Test extraction of complex default values (objects, arrays)."""
        cl = CurriculumLearning()

        # Clear all levels except 1 to ensure we get only our mock task
        for level in range(0, 7):
            if level != 1:
                cl.tasks_by_level[level] = []

        cl.tasks_by_level[1] = [
            {
                "type": "puzzle",
                "language": "javascript",
                "puzzle_name": "ComplexPuzzle",
                "data": {
                    "name": "ComplexPuzzle",
                    "docstring": "Complex defaults",
                    "sat": "function sat(result, config = { a: 1, b: 2 }, items = [1, 2, 3])",
                    "sol": "function sol(config, items) {}",
                    "ans_type": "any",
                    "rating": 1,
                },
                "rating": 1,
                "id": "puzzle_complex",
            }
        ]

        task = cl.get_prompt()
        inputs = task.expected_answer["inputs"]

        # Verify complex values are properly extracted
        self.assertIn("config", inputs)
        self.assertIn("items", inputs)
        self.assertEqual(inputs["config"], {"a": 1, "b": 2})
        self.assertEqual(inputs["items"], [1, 2, 3])

    def test_puzzle_inputs_in_expected_answer_structure(self):
        """Verify expected_answer has correct structure with inputs."""
        cl = CurriculumLearning()

        # Clear all levels except 1 to ensure we get only our mock task
        for level in range(0, 7):
            if level != 1:
                cl.tasks_by_level[level] = []

        cl.tasks_by_level[1] = [
            {
                "type": "puzzle",
                "language": "javascript",
                "puzzle_name": "Test",
                "data": {
                    "name": "Test",
                    "docstring": "Test",
                    "sat": "function sat(result, x = 10)",
                    "sol": "function sol(x) {}",
                    "ans_type": "number",
                    "rating": 1,
                },
                "rating": 1,
                "id": "puzzle_test",
            }
        ]

        task = cl.get_prompt()
        expected_answer = task.expected_answer

        # Verify expected_answer structure
        self.assertIsInstance(expected_answer, dict)
        self.assertIn("puzzle", expected_answer)
        self.assertIn("inputs", expected_answer)
        self.assertIn("language", expected_answer)

        # Verify types
        self.assertIsInstance(expected_answer["puzzle"], str)
        self.assertIsInstance(expected_answer["inputs"], dict)
        self.assertIsInstance(expected_answer["language"], str)


class TestParamExtractorUsageInCurriculum(unittest.TestCase):
    """Test that curriculum.py properly uses param_extractor utility."""

    def test_extract_puzzle_inputs_function_exists_and_is_used(self):
        """Verify curriculum imports and uses extract_puzzle_inputs."""
        # This test verifies the function is importable
        from infinite_rl.curriculum import extract_puzzle_inputs

        # Verify it's callable
        self.assertTrue(callable(extract_puzzle_inputs))

    def test_python_vs_javascript_extraction_parity(self):
        """Verify Python and JavaScript extraction produce equivalent results."""
        python_sat = 'def sat(r, param1="test", param2=42):'
        js_sat = 'function sat(r, param1 = "test", param2 = 42)'

        python_inputs = extract_puzzle_inputs({"sat": python_sat}, "python")
        js_inputs = extract_puzzle_inputs({"sat": js_sat}, "javascript")

        # Both should extract the same parameters with same values
        self.assertEqual(python_inputs, js_inputs)
        self.assertEqual(python_inputs, {"param1": "test", "param2": 42})


if __name__ == "__main__":
    unittest.main()
