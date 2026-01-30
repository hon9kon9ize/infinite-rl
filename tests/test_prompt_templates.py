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
    format_reflective_math_prompt,
    format_reflective_puzzle_prompt,
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

    def test_language_instruction_english(self):
        """Test that English (default) doesn't add language instruction."""
        problem = "Calculate 10 / 2"
        result = format_math_prompt(problem, language="en")

        # English should not add explicit language instruction
        # (though it may still be implicit in the prompt)
        self.assertIn(problem, result)

    def test_language_instruction_chinese(self):
        """Test Chinese language instruction."""
        problem = "計算 5 + 5"
        result = format_math_prompt(problem, language="zh")

        self.assertIn("Mandarin", result)
        self.assertIn(problem, result)

    def test_language_instruction_cantonese(self):
        """Test Cantonese language instruction."""
        problem = "計算 7 + 3"
        result = format_math_prompt(problem, language="yue")

        self.assertIn("Cantonese", result)
        self.assertIn(problem, result)

    def test_language_instruction_unknown(self):
        """Test unknown language falls back to language code."""
        problem = "Resolver: 2 + 2"
        result = format_math_prompt(problem, language="es")

        self.assertIn("es", result)  # Should use the code as fallback
        self.assertIn(problem, result)

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
        self.assertIn("Solve this programming puzzle", result)
        self.assertIn("fibonacci", result)
        self.assertIn("Return the nth Fibonacci number", result)
        self.assertIn("<think>", result)
        self.assertIn("<answer>", result)
        self.assertIn("```python", result)

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
        """Test puzzle with custom think and answer tags."""
        result = format_puzzle_prompt(
            self.puzzle_data, "python", think_tag="analysis", answer_tag="solution"
        )

        self.assertIn("<analysis>", result)
        self.assertIn("</analysis>", result)
        self.assertIn("<solution>", result)
        self.assertIn("</solution>", result)
        self.assertNotIn("<think>", result)
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
        self.assertIn("<answer>", result)


class TestFormatReflectiveMathPrompt(unittest.TestCase):
    """Test cases for format_reflective_math_prompt function."""

    def test_reflective_math_structure(self):
        """Test that reflective math prompt has required sections."""
        original = "What is 2 + 2?"
        previous = "The answer is 4"
        result = format_reflective_math_prompt(original, previous)

        # Should include sections for review
        self.assertIn("Review", result)
        self.assertIn("Original Problem", result)
        self.assertIn("Previous Attempt", result)
        self.assertIn("solve the problem again", result)

    def test_reflective_includes_original_prompt(self):
        """Test that original prompt is included."""
        original = "Calculate 3 * 5"
        previous = "15"
        result = format_reflective_math_prompt(original, previous)

        self.assertIn(original, result)

    def test_reflective_includes_previous_attempt(self):
        """Test that previous attempt is included."""
        original = "What is 10 - 3?"
        previous = "The answer is 7"
        result = format_reflective_math_prompt(original, previous)

        self.assertIn(previous, result)

    def test_reflective_with_empty_previous_attempt(self):
        """Test reflective prompt with no previous attempt."""
        original = "Solve: x + 2 = 5"
        result = format_reflective_math_prompt(original, "")

        self.assertIn("<no output recorded>", result)
        self.assertIn(original, result)

    def test_reflective_custom_tags(self):
        """Test reflective prompt with custom answer/think tags."""
        original = "Calculate 4 / 2"
        previous = "Error: cannot divide"
        result = format_reflective_math_prompt(
            original, previous, answer_tag="result", think_tag="analysis"
        )

        # Should still include the original problem and previous attempt
        self.assertIn(original, result)
        self.assertIn(previous, result)

    def test_reflective_problem_appears_twice(self):
        """Test that original problem appears twice (context + task)."""
        original = "Unique problem statement #12345"
        previous = "Attempt"
        result = format_reflective_math_prompt(original, previous)

        # Should appear at least twice
        count = result.count(original)
        self.assertGreaterEqual(count, 2)


class TestFormatReflectivePuzzlePrompt(unittest.TestCase):
    """Test cases for format_reflective_puzzle_prompt function."""

    def test_reflective_puzzle_structure(self):
        """Test that reflective puzzle prompt has required sections."""
        original = "Solve this function"
        previous = "def sol(): return None"
        result = format_reflective_puzzle_prompt(original, previous, language="python")

        self.assertIn("Review", result)
        self.assertIn("Original Task", result)
        self.assertIn("Previous Attempt", result)
        self.assertIn("solve this puzzle again", result)

    def test_reflective_puzzle_includes_original(self):
        """Test that original prompt is included."""
        original = "Write a function that returns True"
        previous = "def f(): return 1"
        result = format_reflective_puzzle_prompt(original, previous)

        self.assertIn(original, result)

    def test_reflective_puzzle_includes_previous(self):
        """Test that previous attempt is included."""
        original = "Task"
        previous = "def solve(): pass"
        result = format_reflective_puzzle_prompt(original, previous)

        self.assertIn(previous, result)

    def test_reflective_puzzle_python_example(self):
        """Test Python code example in reflective prompt."""
        original = "Write a function"
        previous = "Bad code"
        result = format_reflective_puzzle_prompt(original, previous, language="python")

        # Should have reflective structure
        self.assertIn("Review", result)
        self.assertIn(original, result)

    def test_reflective_puzzle_javascript_example(self):
        """Test JavaScript code example in reflective prompt."""
        original = "Write a function"
        previous = "Bad code"
        result = format_reflective_puzzle_prompt(
            original, previous, language="javascript"
        )

        # Should have reflective structure
        self.assertIn("Review", result)
        self.assertIn(original, result)

    def test_reflective_puzzle_empty_previous(self):
        """Test reflective puzzle with no previous attempt."""
        original = "Implement quicksort"
        result = format_reflective_puzzle_prompt(original, "")

        self.assertIn("<no output recorded>", result)
        self.assertIn(original, result)

    def test_reflective_puzzle_custom_tags(self):
        """Test custom think and answer tags."""
        original = "Write fibonacci"
        previous = "def fib(): return 0"
        result = format_reflective_puzzle_prompt(
            original,
            previous,
            language="python",
            answer_tag="code",
            think_tag="strategy",
        )

        # Should work with custom tags
        self.assertIn(original, result)
        self.assertIn(previous, result)

    def test_reflective_puzzle_problem_appears_twice(self):
        """Test that original problem appears twice."""
        original = "Unique problem #54321"
        previous = "Attempt"
        result = format_reflective_puzzle_prompt(original, previous)

        count = result.count(original)
        self.assertGreaterEqual(count, 2)


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


class TestPromptIntegration(unittest.TestCase):
    """Integration tests for prompt formatting."""

    def test_math_reflective_workflow(self):
        """Test complete math task + reflective workflow."""
        # Initial task
        problem = "What is 5 * 6?"
        initial_prompt = format_math_prompt(problem)
        self.assertIn(problem, initial_prompt)

        # After failure - reflective prompt
        failed_attempt = "I don't know"
        reflective_prompt = format_reflective_math_prompt(problem, failed_attempt)

        self.assertIn(problem, reflective_prompt)
        self.assertIn(failed_attempt, reflective_prompt)
        self.assertIn("Review", reflective_prompt)

    def test_puzzle_reflective_workflow(self):
        """Test complete puzzle task + reflective workflow."""
        puzzle_data = {
            "name": "sum_array",
            "docstring": "Sum all elements in an array",
            "sat": "function sat(a) { return sum([1,2,3]) === 6; }",
            "sol": "function sum(a) { return a.reduce((x,y) => x+y, 0); }",
        }

        # Initial task
        initial_prompt = format_puzzle_prompt(puzzle_data, "javascript")
        self.assertIn("sum_array", initial_prompt)

        # After failure - reflective
        failed_code = "function sum() { return 0; }"
        reflective_prompt = format_reflective_puzzle_prompt(
            initial_prompt, failed_code, language="javascript"
        )

        self.assertIn(failed_code, reflective_prompt)
        self.assertIn("Review", reflective_prompt)

    def test_multilingual_math_workflow(self):
        """Test multilingual math task workflow."""
        for lang_code in ["en", "zh", "yue"]:
            problem = "Calculate 2 + 2"
            prompt = format_math_prompt(problem, language=lang_code)

            self.assertIn(problem, prompt)
            self.assertIn("<think>", prompt)
            self.assertIn("<answer>", prompt)

            if lang_code != "en":
                self.assertIn(LANG_MAP[lang_code], prompt)


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

    def test_previous_attempt_with_tags(self):
        """Test reflective prompt when previous attempt contains XML-like tags."""
        original = "Problem"
        previous = "<answer>5</answer>"  # User put tags in their attempt
        result = format_reflective_math_prompt(original, previous)

        self.assertIn(previous, result)  # Should include it as-is


if __name__ == "__main__":
    unittest.main()
