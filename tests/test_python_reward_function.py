import unittest
import json
from infinite_rl.reward_functions.coding import PythonRewardFunction


class TestPythonRewardFunction(unittest.TestCase):
    """Test Python reward function with Python example."""

    def setUp(self):
        self.reward_fn = PythonRewardFunction(task_name="python")
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
        # Format is validated via separate format reward; correctness is unified in `score`
        self.assertGreaterEqual(float(score.score), 0.0)
        self.assertLessEqual(float(score.score), 1.0)

    def test_missing_code_block(self):
        """Test when code block is missing."""
        model_output = "<answer>This is just plain text, no code block.</answer>"
        expected_output = '{"result": [2, 4, 6, 8]}'

        score = self.reward_fn.compute_reward(model_output, expected_output)
        self.assertEqual(score.score, 0.0)

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

        # Syntax errors result in stderr; formatting (code fence) still valid
        self.assertEqual(score.score, 0.0)

    def test_order_sensitivity_in_string_matching(self):
        """Test that wrong order in string output results in 0.0 correctness under exact match."""
        # Task: Reverse the words
        model_output = """<answer>
```python
print("hello world")
```
</answer>"""
        expected_output = "world hello"

        score = self.reward_fn.compute_reward(model_output, expected_output)

        # Execution ran, but exact correctness must be 0.0
        self.assertEqual(score.score, 0.0)

    def test_json_robustness(self):
        """Test JSON output with different formatting and key order. Under exact-match semantics, these should not be considered equal."""
        # Scenario 1: Key order independence (now treated as mismatch)
        model_output1 = """<answer>
```python
print('{"a": 1, "b": 2}')
```
</answer>"""
        expected_output1 = '{"b": 2, "a": 1}'
        score1 = self.reward_fn.compute_reward(model_output1, expected_output1)
        self.assertEqual(score1.score, 0.0)

        # Scenario 2: Whitespace independence (now treated as mismatch)
        model_output2 = """<answer>
```python
print('{"result"  :   [1, 2, 3]}')
```
</answer>"""
        expected_output2 = '{"result":[1,2,3]}'
        score2 = self.reward_fn.compute_reward(model_output2, expected_output2)
        self.assertEqual(score2.score, 0.0)

    def test_numeric_tolerance(self):
        """Test numeric output with floating point tolerance behavior removed (exact match required)."""
        # Scenario 1: Small difference within 1e-9
        model_output1 = """<answer>
```python
print(3.141592653589)
```
</answer>"""
        expected_output1 = 3.141592653590
        score1 = self.reward_fn.compute_reward(model_output1, expected_output1)
        # Exact-match semantics: different numeric value -> 0.0
        self.assertEqual(score1.score, 0.0)

        # Scenario 2: Larger difference outside tolerance
        model_output2 = """<answer>
```python
print(3.14)
```
</answer>"""
        expected_output2 = 3.14159
        score2 = self.reward_fn.compute_reward(model_output2, expected_output2)
        self.assertEqual(score2.score, 0.0)

    def test_whitespace_normalization(self):
        """Test string matching with extreme whitespace differences. Exact-match semantics -> mismatch."""
        model_output = """<answer>
```python
print("  hello      world\n\n  ")
```
</answer>"""
        expected_output = "hello world"
        score = self.reward_fn.compute_reward(model_output, expected_output)
        self.assertEqual(score.score, 0.0)

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
        self.assertEqual(score.score, 1.0)

    def test_language_specific_extraction(self):
        """Test picking the correct language block when multiple languages are present (Python-specific reward)."""
        model_output = """<answer>
```javascript
console.log("js");
```
```python
print("py");
```
</answer>"""
        # As a Python-specific reward function, it should prefer the python block
        score = self.reward_fn.compute_reward(model_output, "py")
        self.assertEqual(score.score, 1.0)

    def test_empty_output_matching(self):
        """Test cases where the code produces empty output."""
        model_output = """<answer>
```python
pass
```
</answer>"""
        # Scenario 1: Expected is also empty
        score1 = self.reward_fn.compute_reward(model_output, "")
        self.assertEqual(score1.score, 1.0)

        # Scenario 2: Expected is NOT empty
        score2 = self.reward_fn.compute_reward(model_output, "some output")
        self.assertEqual(score2.score, 0.0)

    def test_nested_json_robustness(self):
        """Test deeply nested JSON structure comparison. Exact-match semantics -> mismatch when ordering differs."""
        nested_data = {"a": [1, {"b": 2}], "c": {"d": [3, 4], "e": "f"}}
        model_output = f"""<answer>
```python
import json
print(json.dumps({nested_data}))
```
</answer>"""
        expected_output = json.dumps({"c": {"e": "f", "d": [3, 4]}, "a": [1, {"b": 2}]})
        score = self.reward_fn.compute_reward(model_output, expected_output)
        self.assertEqual(score.score, 0.0)
