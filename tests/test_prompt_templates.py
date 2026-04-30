"""
Unit tests for prompt template formatting functions.

Tests for:
- format_math_prompt: Math problem formatting with reasoning and answer tags
- format_puzzle_prompt: Programming puzzle formatting with code blocks
- format_reflective_math_prompt: Reflective prompts for failed math tasks
- format_reflective_puzzle_prompt: Reflective prompts for failed puzzle tasks
"""

import unittest
from infinite_rl.prompt_templates import (
    format_math_prompt,
    format_puzzle_prompt,
    LANG_MAP,
)


class TestFormatMathPrompt(unittest.TestCase):
    """Test cases for format_math_prompt function."""

    def test_basic_math_prompt_structure(self):
        """Test that math prompt contains required tags and structure."""
        problem = "What is 2 + 2?"
        result = format_math_prompt(problem)

        # Check for required tags
        self.assertIn("<think>", result)
        self.assertIn("</think>", result)
        self.assertIn("<answer>", result)
        self.assertIn("</answer>", result)

        # Check for problem statement
        self.assertIn(problem, result)

        # Check tag order (think before answer)
        think_start = result.index("<think>")
        answer_start = result.index("<answer>")
        self.assertLess(think_start, answer_start)

    def test_custom_answer_tag(self):
        """Test custom answer tag name."""
        problem = "Solve: x + 5 = 10"
        result = format_math_prompt(problem, answer_tag="result")

        self.assertIn("<result>", result)
        self.assertIn("</result>", result)
        self.assertNotIn("<answer>", result)

    def test_custom_think_tag(self):
        """Test custom think tag name."""
        problem = "What is 3 * 4?"
        result = format_math_prompt(problem, think_tag="reasoning")

        self.assertIn("<reasoning>", result)
        self.assertIn("</reasoning>", result)
        self.assertNotIn("<think>", result)

    def test_prompt_instructions_present(self):
        """Test that prompt includes instructions about format."""
        problem = "What is the square root of 16?"
        result = format_math_prompt(problem)

        # Should instruct about numeric result
        self.assertIn("numeric", result.lower())

    def test_empty_problem(self):
        """Test handling of empty problem statement."""
        result = format_math_prompt("")
        self.assertIn("<think>", result)
        self.assertIn("<answer>", result)


class TestFormatPuzzlePrompt(unittest.TestCase):
    """Test cases for format_puzzle_prompt function."""

    def setUp(self):
        """Set up test puzzle data."""
        self.puzzle_data = {
            "name": "fibonacci",
            "docstring": "Return the nth Fibonacci number.",
            "sat": "function sat(n) { return fib(5) === 5; }",
            "sol": "function fib(n) { if (n <= 1) return n; return fib(n-1) + fib(n-2); }",
        }

    def test_basic_puzzle_prompt_structure(self):
        """Test that puzzle prompt contains required sections."""
        result = format_puzzle_prompt(self.puzzle_data, "python")

        # Check for required components
        self.assertIn("# fibonacci", result)
        self.assertIn("Return the nth Fibonacci number", result)
        self.assertIn("<think>", result)
        self.assertIn("```python", result)
        # Puzzle prompts no longer use <answer> tags (code blocks are extracted directly)
        self.assertNotIn("<answer>", result)

    def test_puzzle_with_javascript(self):
        """Test puzzle with JavaScript language."""
        result = format_puzzle_prompt(self.puzzle_data, "javascript")

        self.assertIn("javascript", result)
        self.assertIn("```javascript", result)
        self.assertIn(self.puzzle_data["sat"], result)

    def test_puzzle_with_python(self):
        """Test puzzle with Python language."""
        result = format_puzzle_prompt(self.puzzle_data, "python")

        self.assertIn("python", result)
        self.assertIn("```python", result)

    def test_custom_puzzle_tags(self):
        """Test puzzle with custom think tag (answer_tag ignored for puzzles)."""
        result = format_puzzle_prompt(
            self.puzzle_data, "python", think_tag="analysis", answer_tag="solution"
        )

        self.assertIn("<analysis>", result)
        self.assertIn("</analysis>", result)
        # Puzzle prompts no longer use answer tags
        self.assertNotIn("<solution>", result)
        self.assertNotIn("<answer>", result)

    def test_puzzle_includes_sat_function(self):
        """Test that SAT function is included in prompt."""
        result = format_puzzle_prompt(self.puzzle_data, "python")

        self.assertIn(self.puzzle_data["sat"], result)

    def test_puzzle_includes_solution_signature(self):
        """Test that solution signature is included."""
        result = format_puzzle_prompt(self.puzzle_data, "python")

        self.assertIn(self.puzzle_data["sol"], result)

    def test_puzzle_code_block_format(self):
        """Test that code blocks are properly formatted."""
        result = format_puzzle_prompt(self.puzzle_data, "python")

        # Should have opening and closing code blocks
        self.assertTrue(result.count("```python") >= 1)
        self.assertTrue(result.count("```") >= 2)  # At least one pair

    def test_puzzle_missing_optional_fields(self):
        """Test handling of missing optional fields."""
        minimal_puzzle = {
            "name": "test_puzzle",
            "docstring": "A simple test",
            "sat": "function sat() { return true; }",
            "sol": "function sol() { return true; }",
        }

        result = format_puzzle_prompt(minimal_puzzle, "python")

        self.assertIn("test_puzzle", result)
        self.assertIn("A simple test", result)

    def test_puzzle_empty_fields(self):
        """Test handling when puzzle data has empty string fields."""
        empty_puzzle = {
            "name": "empty",
            "docstring": "",
            "sat": "",
            "sol": "",
        }

        result = format_puzzle_prompt(empty_puzzle, "python")

        self.assertIn("empty", result)
        self.assertIn("<think>", result)
        # Puzzle prompts no longer use <answer> tags
        self.assertNotIn("<answer>", result)


class TestLanguageMapping(unittest.TestCase):
    """Test cases for language code mappings."""

    def test_lang_map_contains_required_keys(self):
        """Test that LANG_MAP has required language codes."""
        self.assertIn("en", LANG_MAP)
        self.assertIn("zh", LANG_MAP)
        self.assertIn("yue", LANG_MAP)

    def test_lang_map_values_are_strings(self):
        """Test that all language names are strings."""
        for code, name in LANG_MAP.items():
            self.assertIsInstance(name, str)
            self.assertGreater(len(name), 0)

    def test_english_mapping(self):
        """Test English language mapping."""
        self.assertEqual(LANG_MAP["en"], "English")

    def test_chinese_mapping(self):
        """Test Mandarin Chinese mapping."""
        self.assertEqual(LANG_MAP["zh"], "Mandarin")

    def test_cantonese_mapping(self):
        """Test Cantonese mapping."""
        self.assertEqual(LANG_MAP["yue"], "Cantonese")


class TestPromptEdgeCases(unittest.TestCase):
    """Test edge cases and error handling."""

    def test_very_long_problem_statement(self):
        """Test handling of very long problem statements."""
        long_problem = "A" * 1000
        result = format_math_prompt(long_problem)

        self.assertIn(long_problem, result)
        self.assertIn("<think>", result)
        self.assertIn("<answer>", result)

    def test_problem_with_special_characters(self):
        """Test problems with special characters."""
        special_problem = "Solve: ∑(x²) + ∫(y) = π"
        result = format_math_prompt(special_problem)

        self.assertIn(special_problem, result)

    def test_problem_with_code_blocks(self):
        """Test problem that contains code blocks."""
        problem_with_code = "Evaluate: `2 + 2` in Python"
        result = format_math_prompt(problem_with_code)

        self.assertIn(problem_with_code, result)

    def test_puzzle_with_multiline_docstring(self):
        """Test puzzle with multi-line docstring."""
        puzzle_data = {
            "name": "complex",
            "docstring": "This is a complex puzzle\nwith multiple lines\nof documentation",
            "sat": "function sat() { return true; }",
            "sol": "function sol() { return true; }",
        }

        result = format_puzzle_prompt(puzzle_data, "python")

        self.assertIn("complex", result)
        self.assertIn("multiple lines", result)


class TestReasoningTemplateMathPrompt(unittest.TestCase):
    """Test format_math_prompt with reasoning_template flag."""

    def test_reasoning_template_omits_think_tag_instructions(self):
        """When reasoning_template=True, prompt should not mention think tags."""
        problem = "What is 2 + 2?"
        result = format_math_prompt(problem, reasoning_template=True)

        # Should NOT have think tag instructions
        self.assertNotIn("<think>", result)
        self.assertNotIn("</think>", result)
        self.assertNotIn("Reasoning", result)

        # Should still have answer tag
        self.assertIn("<answer>", result)
        self.assertIn("</answer>", result)
        self.assertIn(problem, result)

    def test_reasoning_template_default_includes_think_tag(self):
        """Default (reasoning_template=False) should include think tag instructions."""
        problem = "What is 2 + 2?"
        result = format_math_prompt(problem)

        self.assertIn("<think>", result)
        self.assertIn("</think>", result)
        self.assertIn("Reasoning", result)
        self.assertIn("<answer>", result)

    def test_reasoning_template_explicit_false(self):
        """Explicit reasoning_template=False should include think tag instructions."""
        problem = "What is 2 + 2?"
        result = format_math_prompt(problem, reasoning_template=False)

        self.assertIn("<think>", result)
        self.assertIn("<answer>", result)


class TestReasoningTemplatePuzzlePrompt(unittest.TestCase):
    """Test format_puzzle_prompt with reasoning_template flag."""

    def setUp(self):
        self.puzzle_data = {
            "name": "fibonacci",
            "docstring": "Return the nth Fibonacci number.",
            "sat": "def sat(n): pass",
            "sol": "def fib(n): pass",
        }

    def test_reasoning_template_omits_think_tag_instructions(self):
        """When reasoning_template=True, prompt should not mention think tags."""
        result = format_puzzle_prompt(
            self.puzzle_data, "python", reasoning_template=True
        )

        # Should NOT have think tag instructions
        self.assertNotIn("<think>", result)
        self.assertNotIn("</think>", result)
        self.assertNotIn("reasoning and approach", result)

        # Should still have code block (no <answer> tags for puzzles)
        self.assertIn("```python", result)
        self.assertIn("fibonacci", result)
        self.assertNotIn("<answer>", result)

    def test_reasoning_template_default_includes_think_tag(self):
        """Default (reasoning_template=False) should include think tag instructions."""
        result = format_puzzle_prompt(self.puzzle_data, "python")

        self.assertIn("<think>", result)
        # Puzzle prompts no longer use <answer> tags
        self.assertNotIn("<answer>", result)

    def test_reasoning_template_with_javascript(self):
        """reasoning_template=True should work with JavaScript too."""
        result = format_puzzle_prompt(
            self.puzzle_data, "javascript", reasoning_template=True
        )

        self.assertNotIn("<think>", result)
        # Puzzle prompts no longer use <answer> tags
        self.assertNotIn("<answer>", result)
        self.assertIn("```javascript", result)


if __name__ == "__main__":
    unittest.main()
