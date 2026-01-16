"""
Unit tests for reward functions using example data.
Tests cover all reward function types with mocked dependencies.
"""

import unittest
import json
from pathlib import Path
from infinite_rl.reward_functions.coding import CodingRewardFunction
from infinite_rl.reward_functions.math import MathRewardFunction
from infinite_rl.parser import ExampleParser


class TestExamples(unittest.TestCase):
    """Data-driven tests using example markdown files."""

    @classmethod
    def setUpClass(cls):
        # The examples directory is now inside the package
        cls.examples_dir = Path(__file__).parent.parent / "infinite_rl" / "examples"
        if not cls.examples_dir.exists():
            # Fallback for localized testing
            cls.examples_dir = Path(__file__).parent.parent / "examples"
        cls.examples = ExampleParser.get_all_examples(cls.examples_dir)

    def test_python_example(self):
        example = self.examples.get("PYTHON")
        self.assertIsNotNone(example)

        reward_fn = CodingRewardFunction(task_name="python")
        reward_fn.initialize()

        # Python example already outputs JSON
        score = reward_fn.compute_reward(example["response"], example["answer"])
        self.assertEqual(score.format_score, 1.0)
        self.assertEqual(score.correctness_score, 1.0)

    def test_javascript_example(self):
        example = self.examples.get("JAVASCRIPT")
        self.assertIsNotNone(example)

        reward_fn = CodingRewardFunction(task_name="javascript")
        reward_fn.set_language("javascript")
        reward_fn.initialize()

        # JS example already outputs JSON
        score = reward_fn.compute_reward(example["response"], example["answer"])
        self.assertEqual(score.format_score, 1.0)
        # Check if the score is 1.0, but if not, print debug
        if score.correctness_score < 1.0:
            print(f"JS Debug: score={score}")
        reward_fn.initialize()

        score = reward_fn.compute_reward(example["response"], example["answer"])
        self.assertEqual(score.format_score, 1.0)
        self.assertEqual(score.correctness_score, 1.0)

    def test_math_example(self):
        example = self.examples.get("MATH")
        self.assertIsNotNone(example)

        reward_fn = MathRewardFunction(task_name="math")
        reward_fn.initialize()

        score = reward_fn.compute_reward(example["response"], example["answer"])
        self.assertEqual(score.format_score, 1.0)
        self.assertEqual(score.correctness_score, 1.0)


class TestCodingRewardFunction(unittest.TestCase):
    """Test coding reward function with Python example."""

    def setUp(self):
        self.reward_fn = CodingRewardFunction(task_name="python_coding")
        self.reward_fn.initialize()

    def test_valid_python_code_with_json_output(self):
        """Test valid Python code with correct JSON output."""
        model_output = """<answer>
```python
import json
def filter_even(numbers):
    return [n for n in numbers if n % 2 == 0]

result = filter_even([1, 2, 3, 4, 5, 6, 7, 8])
print(json.dumps({"result": result}))
```
</answer>"""
        expected_output = '{"result": [2, 4, 6, 8]}'

        score = self.reward_fn.compute_reward(model_output, expected_output)

        self.assertIsNotNone(score)
        self.assertGreaterEqual(score.format_score, 0.0)
        self.assertLessEqual(score.format_score, 1.0)
        self.assertGreaterEqual(score.correctness_score, 0.0)
        self.assertLessEqual(score.correctness_score, 1.0)

    def test_missing_code_block(self):
        """Test when code block is missing."""
        model_output = "<answer>This is just plain text, no code block.</answer>"
        expected_output = '{"result": [2, 4, 6, 8]}'

        score = self.reward_fn.compute_reward(model_output, expected_output)

        self.assertEqual(score.format_score, 0.5)
        self.assertEqual(score.correctness_score, 0.0)

    def test_custom_tag_for_code_block(self):
        """Test that code block inside a custom tag can be found and executed."""
        model_output = "<final>```python\nprint(2+2)\n```</final>"
        expected_output = "4"

        score = self.reward_fn.compute_reward(
            model_output, expected_output, answer_tag="final"
        )

        self.assertEqual(score.format_score, 1.0)
        self.assertEqual(score.correctness_score, 1.0)

    def test_syntax_error_in_code(self):
        """Test code with syntax errors."""
        model_output = """<answer>
```python
def filter_even(numbers)  # Missing colon
    return [n for n in numbers if n % 2 == 0]
```
</answer>"""
        expected_output = '{"result": [2, 4, 6, 8]}'

        score = self.reward_fn.compute_reward(model_output, expected_output)

        # Syntax errors result in stderr.
        # Format score is 0.5 because the code block was successfully extracted.
        self.assertEqual(score.format_score, 0.5)
        self.assertEqual(score.correctness_score, 0.0)

    def test_order_sensitivity_in_string_matching(self):
        """Test that wrong order in string output results in 0.0 correctness."""
        # Task: Reverse the words
        model_output = """<answer>
```python
print("hello world")
```
</answer>"""
        expected_output = "world hello"

        score = self.reward_fn.compute_reward(model_output, expected_output)

        # It should have 1.0 format (executed fine) but low correctness (wrong order)
        self.assertEqual(score.format_score, 1.0)
        self.assertLess(
            score.correctness_score, 0.6
        )  # Sequence ratio will be relatively low
        self.assertGreater(score.correctness_score, 0.0)

    def test_json_robustness(self):
        """Test JSON output with different formatting and key order."""
        # Scenario 1: Key order independence
        model_output1 = """<answer>
```python
print('{"a": 1, "b": 2}')
```
</answer>"""
        expected_output1 = '{"b": 2, "a": 1}'
        score1 = self.reward_fn.compute_reward(model_output1, expected_output1)
        self.assertEqual(score1.correctness_score, 1.0)

        # Scenario 2: Whitespace independence
        model_output2 = """<answer>
```python
print('{"result"  :   [1, 2, 3]}')
```
</answer>"""
        expected_output2 = '{"result":[1,2,3]}'
        score2 = self.reward_fn.compute_reward(model_output2, expected_output2)
        self.assertEqual(score2.correctness_score, 1.0)

    def test_numeric_tolerance(self):
        """Test numeric output with floating point tolerance."""
        # Scenario 1: Small difference within 1e-9
        model_output1 = """<answer>
```python
print(3.141592653589)
```
</answer>"""
        expected_output1 = 3.141592653590
        score1 = self.reward_fn.compute_reward(model_output1, expected_output1)
        # Similarity should be 0.99 for small numeric differences
        self.assertEqual(score1.correctness_score, 0.99)

        # Scenario 2: Larger difference outside tolerance
        model_output2 = """<answer>
```python
print(3.14)
```
</answer>"""
        expected_output2 = 3.14159
        score2 = self.reward_fn.compute_reward(model_output2, expected_output2)
        # Ratio will be partial but higher than 0.0
        self.assertLess(score2.correctness_score, 0.99)
        self.assertGreater(score2.correctness_score, 0.5)

    def test_whitespace_normalization(self):
        """Test string matching with extreme whitespace differences."""
        model_output = """<answer>
```python
print("  hello      world\\n\\n  ")
```
</answer>"""
        expected_output = "hello world"
        score = self.reward_fn.compute_reward(model_output, expected_output)
        # Normalized whitespace comparison should return 0.95 similarity
        self.assertEqual(score.correctness_score, 0.95)

    def test_multiple_code_blocks(self):
        """Test that the reward function correctly extracts code from multiple blocks."""
        # 1. First block is picked if no language specified or first matches
        model_output = """<answer>
Some initial thoughts.
```python
print("first")
```
Some other code.
```python
print("second")
```
</answer>"""
        # The current implementation uses re.search, which finds the FIRST match.
        score = self.reward_fn.compute_reward(model_output, "first")
        self.assertEqual(score.correctness_score, 1.0)

    def test_language_specific_extraction(self):
        """Test picking the correct language block when multiple languages are present."""
        self.reward_fn.set_language("python")
        model_output = """<answer>
```javascript
console.log("js");
```
```python
print("py");
```
</answer>"""
        # It should prefer the python block
        score = self.reward_fn.compute_reward(model_output, "py")
        self.assertEqual(score.correctness_score, 1.0)

    def test_empty_output_matching(self):
        """Test cases where the code produces empty output."""
        model_output = """<answer>
```python
pass
```
</answer>"""
        # Scenario 1: Expected is also empty
        score1 = self.reward_fn.compute_reward(model_output, "")
        self.assertEqual(score1.correctness_score, 1.0)

        # Scenario 2: Expected is NOT empty
        score2 = self.reward_fn.compute_reward(model_output, "some output")
        self.assertEqual(score2.correctness_score, 0.0)

    def test_nested_json_robustness(self):
        """Test deeply nested JSON structure comparison."""
        nested_data = {"a": [1, {"b": 2}], "c": {"d": [3, 4], "e": "f"}}
        model_output = f"""<answer>
```python
import json
print(json.dumps({nested_data}))
```
</answer>"""
        expected_output = json.dumps({"c": {"e": "f", "d": [3, 4]}, "a": [1, {"b": 2}]})
        score = self.reward_fn.compute_reward(model_output, expected_output)
        self.assertEqual(score.correctness_score, 1.0)


class TestMathRewardFunction(unittest.TestCase):
    """Test math reward function with SymPy evaluation."""

    def setUp(self):
        self.reward_fn = MathRewardFunction(task_name="math")
        self.reward_fn.initialize()

    def test_correct_integral_answer(self):
        """Test correct mathematical answer."""
        model_output = """
The integral of f(x) = 2x^3 - 4x + 1:

Using the power rule:
= 2x^4/4 - 4x^2/2 + x + C
= (1/2)x^4 - 2x^2 + x + C

<answer>(1/2)x^4 - 2x^2 + x + C</answer>
"""
        expected_output = "(1/2)x^4 - 2x^2 + x + C"

        score = self.reward_fn.compute_reward(model_output, expected_output)

        self.assertEqual(score.format_score, 1.0)  # Answer tag found
        self.assertEqual(score.correctness_score, 1.0)  # Mathematically correct

    def test_latex_answer_symbolic_match(self):
        """Latex formatted answer should match symbolic expected output."""
        model_output = """
The integral of f(x) = 2x^3 - 4x + 1 (LaTeX form):

Using the power rule:
= 2x^4/4 - 4x^2/2 + x + C
= \frac{1}{2}x^4 - 2x^2 + x + C

<answer>\\frac{1}{2}x^4 - 2x^2 + x + C</answer>
"""
        expected_output = "(1/2)x^4 - 2x^2 + x + C"

        score = self.reward_fn.compute_reward(model_output, expected_output)

        self.assertEqual(score.format_score, 1.0)
        self.assertEqual(score.correctness_score, 1.0)

    def test_missing_answer_tag(self):
        """Test when answer tag is missing."""
        model_output = "The answer is x^2 + 2x + 1 but no tags here."
        expected_output = "x^2 + 2x + 1"

        score = self.reward_fn.compute_reward(model_output, expected_output)

        self.assertEqual(score.format_score, 0.0)
        self.assertEqual(score.correctness_score, 0.0)

    def test_incorrect_mathematical_answer(self):
        """Test mathematically incorrect answer."""
        model_output = "<answer>x^4 - 2x^2 + x + C</answer>"
        expected_output = "(1/2)x^4 - 2x^2 + x + C"

        score = self.reward_fn.compute_reward(model_output, expected_output)

        self.assertEqual(score.format_score, 1.0)
        self.assertEqual(score.correctness_score, 0.0)

    def test_integer_expected_output(self):
        """Test with integer expected output."""
        model_output = "<answer>42</answer>"
        expected_output = 42

        score = self.reward_fn.compute_reward(model_output, expected_output)

        self.assertEqual(score.format_score, 1.0)
        self.assertEqual(score.correctness_score, 1.0)

    def test_numeric_only_comparison_from_string(self):
        """Ensure math reward compares numbers only when annotations are present."""
        cases = [
            ("<answer>10 \\text{hours}</answer>", 10),
            ("<answer>20\\,\\text{ml}</answer>", 20),
            ("<answer>50 \\text{ml/dose}</answer>", 50),
            ("<answer>$50</answer>", 50),
            ("<answer>$1000$</answer>", 1000),
            ("<answer>$1000</answer>", 1000),
            ("<answer>12 \\, \\text{cm|kg\\}</answer>", 12),
            ("<answer>3.14159</answer>", 3.14159),
        ]
        for model_out, expected in cases:
            score = self.reward_fn.compute_reward(model_out, expected)
            self.assertEqual(score.format_score, 1.0)
            self.assertEqual(score.correctness_score, 1.0)

    def test_non_numeric_fails(self):
        """If the answer can't be parsed to a number, correctness is 0.0."""
        score = self.reward_fn.compute_reward("<answer>ten hours</answer>", 10)
        self.assertEqual(score.format_score, 1.0)
        self.assertEqual(score.correctness_score, 0.0)

    def test_integer_with_wrong_value(self):
        """Test with integer but wrong value (now strict numeric equality)."""
        model_output = "<answer>40</answer>"
        expected_output = 42

        score = self.reward_fn.compute_reward(model_output, expected_output)

        self.assertEqual(score.format_score, 1.0)
        self.assertEqual(score.correctness_score, 0.0)

    def test_custom_tag_numeric_extraction(self):
        """Test numeric extraction when model uses a custom tag name."""
        model_output = "<result>100</result>"
        expected_output = 100

        score = self.reward_fn.compute_reward(
            model_output, expected_output, answer_tag="result"
        )

        self.assertEqual(score.format_score, 1.0)
        self.assertEqual(score.correctness_score, 1.0)


if __name__ == "__main__":
    unittest.main()
