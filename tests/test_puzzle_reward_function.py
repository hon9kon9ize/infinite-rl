import unittest
from infinite_rl.reward_functions.puzzle import PuzzleRewardFunction


class TestPuzzleRewardFunction(unittest.TestCase):
    """Test Puzzle reward function with puzzle examples."""

    def setUp(self):
        self.reward_fn = PuzzleRewardFunction(task_name="puzzle", language="javascript")
        self.reward_fn.initialize()

    def test_valid_js_puzzle_solution(self):
        model_output = """<answer>
```javascript
function sol(inputs) {
    return "19";
}
```
</answer>"""
        expected_output = {
            "puzzle": "SumOfDigits",
            "inputs": {"s": 10},
            "language": "javascript",
        }
        score = self.reward_fn.compute_reward(model_output, expected_output)
        self.assertEqual(score.score, 1.0)

    def test_invalid_js_puzzle_solution(self):
        model_output = """<answer>
```javascript
function sol(inputs) {
    return "18";  // Wrong answer
}
```
</answer>"""
        expected_output = {
            "puzzle": "SumOfDigits",
            "inputs": {"s": 10},
            "language": "javascript",
        }
        score = self.reward_fn.compute_reward(model_output, expected_output)
        self.assertEqual(score.score, 0.0)

    def test_missing_code_block(self):
        model_output = "<answer>This is just plain text, no code block.</answer>"
        expected_output = {
            "puzzle": "SumOfDigits",
            "inputs": {"s": 10},
            "language": "javascript",
        }
        score = self.reward_fn.compute_reward(model_output, expected_output)
        self.assertEqual(score.score, 0.0)

    def test_missing_sol_function(self):
        model_output = """<answer>
```javascript
console.log("no sol function");
```
</answer>"""
        expected_output = {
            "puzzle": "SumOfDigits",
            "inputs": {"s": 10},
            "language": "javascript",
        }
        score = self.reward_fn.compute_reward(model_output, expected_output)
        self.assertEqual(score.score, 0.0)

    def test_invalid_expected_output(self):
        model_output = """<answer>
```javascript
function sol(inputs) { return "19"; }
```
</answer>"""
        expected_output = "invalid json"
        score = self.reward_fn.compute_reward(model_output, expected_output)
        self.assertEqual(score.score, 0.0)

    def test_missing_puzzle_name(self):
        model_output = """<answer>
```javascript
function sol(inputs) { return "19"; }
```
</answer>"""
        expected_output = {"inputs": {"s": 10}, "language": "javascript"}
        score = self.reward_fn.compute_reward(model_output, expected_output)
        self.assertEqual(score.score, 0.0)

    def test_valid_js_quadratic_root(self):
        model_output = """<answer>
```javascript
function sol(inputs) {
    return 1.0;
}
```
</answer>"""
        expected_output = {
            "puzzle": "QuadraticRoot",
            "inputs": {"coeffs": [1.0, -3.0, 2.0]},
            "language": "javascript",
        }
        score = self.reward_fn.compute_reward(model_output, expected_output)
        self.assertEqual(score.score, 1.0)

    def test_invalid_js_quadratic_root(self):
        model_output = """<answer>
```javascript
function sol(inputs) {
    return 5.0;  // Not a root
}
```
</answer>"""
        expected_output = {
            "puzzle": "QuadraticRoot",
            "inputs": {"coeffs": [1.0, -3.0, 2.0]},
            "language": "javascript",
        }
        score = self.reward_fn.compute_reward(model_output, expected_output)
        self.assertEqual(score.score, 0.0)

    def test_valid_js_all_quadratic_roots(self):
        model_output = """<answer>
```javascript
function sol(inputs) {
    return [1.0, 2.0];
}
```
</answer>"""
        expected_output = {
            "puzzle": "AllQuadraticRoots",
            "inputs": {"coeffs": [-3.0, 2.0]},
            "language": "javascript",
        }
        score = self.reward_fn.compute_reward(model_output, expected_output)
        self.assertEqual(score.score, 1.0)

    def test_invalid_js_all_quadratic_roots(self):
        model_output = """<answer>
```javascript
function sol(inputs) {
    return [1.0, 3.0];  // Wrong roots
}
```
</answer>"""
        expected_output = {
            "puzzle": "AllQuadraticRoots",
            "inputs": {"coeffs": [-3.0, 2.0]},
            "language": "javascript",
        }
        score = self.reward_fn.compute_reward(model_output, expected_output)
        self.assertEqual(score.score, 0.0)

    def test_valid_js_float_decimal(self):
        model_output = """<answer>
```javascript
function sol(inputs) {
    return 0.5;
}
```
</answer>"""
        expected_output = {
            "puzzle": "FloatWithDecimalValue",
            "inputs": {"v": 5, "d": 0.1},
            "language": "javascript",
        }
        score = self.reward_fn.compute_reward(model_output, expected_output)
        self.assertEqual(score.score, 1.0)


class TestPuzzleRewardFunctionPython(unittest.TestCase):
    """Test Puzzle reward function with Python puzzle examples."""

    def setUp(self):
        self.reward_fn = PuzzleRewardFunction(task_name="puzzle", language="python")
        self.reward_fn.initialize()

    def test_valid_python_puzzle_solution(self):
        model_output = """<answer>
```python
def sol(inputs):
    return "19"
```
</answer>"""
        expected_output = {
            "puzzle": "SumOfDigits",
            "inputs": {"s": 10},
            "language": "python",
        }
        score = self.reward_fn.compute_reward(model_output, expected_output)
        self.assertEqual(score.score, 1.0)

    def test_invalid_python_puzzle_solution(self):
        model_output = """<answer>
```python
def sol(inputs):
    return "18"  # Wrong answer
```
</answer>"""
        expected_output = {
            "puzzle": "SumOfDigits",
            "inputs": {"s": 10},
            "language": "python",
        }
        score = self.reward_fn.compute_reward(model_output, expected_output)
        self.assertEqual(score.score, 0.0)

    def test_missing_code_block_python(self):
        model_output = "<answer>This is just plain text, no code block.</answer>"
        expected_output = {
            "puzzle": "SumOfDigits",
            "inputs": {"s": 10},
            "language": "python",
        }
        score = self.reward_fn.compute_reward(model_output, expected_output)
        self.assertEqual(score.score, 0.0)

    def test_missing_sol_function_python(self):
        model_output = """<answer>
```python
print("no sol function")
```
</answer>"""
        expected_output = {
            "puzzle": "SumOfDigits",
            "inputs": {"s": 10},
            "language": "python",
        }
        score = self.reward_fn.compute_reward(model_output, expected_output)
        self.assertEqual(score.score, 0.0)

    def test_valid_python_quadratic_root(self):
        model_output = """<answer>
```python
def sol(inputs):
    return 1.0
```
</answer>"""
        expected_output = {
            "puzzle": "QuadraticRoot",
            "inputs": {"coeffs": [1.0, -3.0, 2.0]},
            "language": "python",
        }
        score = self.reward_fn.compute_reward(model_output, expected_output)
        self.assertEqual(score.score, 1.0)

    def test_invalid_python_quadratic_root(self):
        model_output = """<answer>
```python
def sol(inputs):
    return 5.0  # Not a root
```
</answer>"""
        expected_output = {
            "puzzle": "QuadraticRoot",
            "inputs": {"coeffs": [1.0, -3.0, 2.0]},
            "language": "python",
        }
        score = self.reward_fn.compute_reward(model_output, expected_output)
        self.assertEqual(score.score, 0.0)

    def test_valid_python_all_quadratic_roots(self):
        model_output = """<answer>
```python
def sol(inputs):
    return [1.0, 2.0]
```
</answer>"""
        expected_output = {
            "puzzle": "AllQuadraticRoots",
            "inputs": {"coeffs": [-3.0, 2.0]},
            "language": "python",
        }
        score = self.reward_fn.compute_reward(model_output, expected_output)
        self.assertEqual(score.score, 1.0)

    def test_invalid_python_all_quadratic_roots(self):
        model_output = """<answer>
```python
def sol(inputs):
    return [1.0, 3.0]  # Wrong roots
```
</answer>"""
        expected_output = {
            "puzzle": "AllQuadraticRoots",
            "inputs": {"coeffs": [-3.0, 2.0]},
            "language": "python",
        }
        score = self.reward_fn.compute_reward(model_output, expected_output)
        self.assertEqual(score.score, 0.0)

    def test_valid_python_float_decimal(self):
        model_output = """<answer>
```python
def sol(inputs):
    return 0.5
```
</answer>"""
        expected_output = {
            "puzzle": "FloatWithDecimalValue",
            "inputs": {"v": 5, "d": 0.1},
            "language": "python",
        }
        score = self.reward_fn.compute_reward(model_output, expected_output)
        self.assertEqual(score.score, 1.0)


if __name__ == "__main__":
    unittest.main()
