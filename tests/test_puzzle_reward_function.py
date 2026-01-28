import unittest
from infinite_rl.reward_functions.puzzle import PuzzleRewardFunction
from infinite_rl.task import Task


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
        task = Task(
            task_id="test_1",
            task_name="test",
            task_type="puzzle",
            level=1,
            prompt="Test",
            expected_answer=expected_output,
            language="javascript",
            model_output=model_output,
        )
        score = self.reward_fn.compute_reward(task)
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
        task = Task(
            task_id="test_2",
            task_name="test",
            task_type="puzzle",
            level=1,
            prompt="Test",
            expected_answer=expected_output,
            language="javascript",
            model_output=model_output,
        )
        score = self.reward_fn.compute_reward(task)
        self.assertEqual(score.score, 0.0)

    def test_missing_code_block(self):
        model_output = "<answer>This is just plain text, no code block.</answer>"
        expected_output = {
            "puzzle": "SumOfDigits",
            "inputs": {"s": 10},
            "language": "javascript",
        }
        task = Task(
            task_id="test_3",
            task_name="test",
            task_type="puzzle",
            level=1,
            prompt="Test",
            expected_answer=expected_output,
            language="javascript",
            model_output=model_output,
        )
        score = self.reward_fn.compute_reward(task)
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
        task = Task(
            task_id="test_4",
            task_name="test",
            task_type="puzzle",
            level=1,
            prompt="Test",
            expected_answer=expected_output,
            language="javascript",
            model_output=model_output,
        )
        score = self.reward_fn.compute_reward(task)
        self.assertEqual(score.score, 0.0)

    def test_invalid_expected_output(self):
        model_output = """<answer>
```javascript
function sol(inputs) { return "19"; }
```
</answer>"""
        task = Task(
            task_id="test_5",
            task_name="test",
            task_type="puzzle",
            level=1,
            prompt="Test",
            expected_answer="invalid json",
            language="javascript",
            model_output=model_output,
        )
        score = self.reward_fn.compute_reward(task)
        self.assertEqual(score.score, 0.0)

    def test_missing_puzzle_name(self):
        model_output = """<answer>
```javascript
function sol(inputs) { return "19"; }
```
</answer>"""
        expected_output = {"inputs": {"s": 10}, "language": "javascript"}
        task = Task(
            task_id="test_6",
            task_name="test",
            task_type="puzzle",
            level=1,
            prompt="Test",
            expected_answer=expected_output,
            language="javascript",
            model_output=model_output,
        )
        score = self.reward_fn.compute_reward(task)
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
        task = Task(
            task_id="test_7",
            task_name="test",
            task_type="puzzle",
            level=1,
            prompt="Test",
            expected_answer=expected_output,
            language="javascript",
            model_output=model_output,
        )
        score = self.reward_fn.compute_reward(task)
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
        task = Task(
            task_id="test_8",
            task_name="test",
            task_type="puzzle",
            level=1,
            prompt="Test",
            expected_answer=expected_output,
            language="javascript",
            model_output=model_output,
        )
        score = self.reward_fn.compute_reward(task)
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
        task = Task(
            task_id="test_9",
            task_name="test",
            task_type="puzzle",
            level=1,
            prompt="Test",
            expected_answer=expected_output,
            language="javascript",
            model_output=model_output,
        )
        score = self.reward_fn.compute_reward(task)
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
        task = Task(
            task_id="test_10",
            task_name="test",
            task_type="puzzle",
            level=1,
            prompt="Test",
            expected_answer=expected_output,
            language="javascript",
            model_output=model_output,
        )
        score = self.reward_fn.compute_reward(task)
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
        task = Task(
            task_id="test_11",
            task_name="test",
            task_type="puzzle",
            level=1,
            prompt="Test",
            expected_answer=expected_output,
            language="javascript",
            model_output=model_output,
        )
        score = self.reward_fn.compute_reward(task)
        self.assertEqual(score.score, 1.0)


class TestPuzzleRewardFunctionPython(unittest.TestCase):
    """Test Puzzle reward function with Python puzzle examples."""

    def setUp(self):
        self.reward_fn = PuzzleRewardFunction(task_name="puzzle", language="python")
        self.reward_fn.initialize()

    def test_valid_python_puzzle_solution(self):
        model_output = """<answer>
```python
def sol(s):
    return "19"
```
</answer>"""
        expected_output = {
            "puzzle": "SumOfDigits",
            "inputs": {"s": 10},
            "language": "python",
        }
        task = Task(
            task_id="test_12",
            task_name="test",
            task_type="puzzle",
            level=1,
            prompt="Test",
            expected_answer=expected_output,
            language="python",
            model_output=model_output,
        )
        score = self.reward_fn.compute_reward(task)
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
        task = Task(
            task_id="test_13",
            task_name="test",
            task_type="puzzle",
            level=1,
            prompt="Test",
            expected_answer=expected_output,
            language="python",
            model_output=model_output,
        )
        score = self.reward_fn.compute_reward(task)
        self.assertEqual(score.score, 0.0)

    def test_missing_code_block_python(self):
        model_output = "<answer>This is just plain text, no code block.</answer>"
        expected_output = {
            "puzzle": "SumOfDigits",
            "inputs": {"s": 10},
            "language": "python",
        }
        task = Task(
            task_id="test_14",
            task_name="test",
            task_type="puzzle",
            level=1,
            prompt="Test",
            expected_answer=expected_output,
            language="python",
            model_output=model_output,
        )
        score = self.reward_fn.compute_reward(task)
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
        task = Task(
            task_id="test_15",
            task_name="test",
            task_type="puzzle",
            level=1,
            prompt="Test",
            expected_answer=expected_output,
            language="python",
            model_output=model_output,
        )
        score = self.reward_fn.compute_reward(task)
        self.assertEqual(score.score, 0.0)

    def test_valid_python_quadratic_root(self):
        model_output = """<answer>
```python
def sol(coeffs):
    return 1.0
```
</answer>"""
        expected_output = {
            "puzzle": "QuadraticRoot",
            "inputs": {"coeffs": [1.0, -3.0, 2.0]},
            "language": "python",
        }
        task = Task(
            task_id="test_16",
            task_name="test",
            task_type="puzzle",
            level=1,
            prompt="Test",
            expected_answer=expected_output,
            language="python",
            model_output=model_output,
        )
        score = self.reward_fn.compute_reward(task)
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
        task = Task(
            task_id="test_17",
            task_name="test",
            task_type="puzzle",
            level=1,
            prompt="Test",
            expected_answer=expected_output,
            language="python",
            model_output=model_output,
        )
        score = self.reward_fn.compute_reward(task)
        self.assertEqual(score.score, 0.0)

    def test_valid_python_all_quadratic_roots(self):
        model_output = """<answer>
```python
def sol(coeffs):
    return [1.0, 2.0]
```
</answer>"""
        expected_output = {
            "puzzle": "AllQuadraticRoots",
            "inputs": {"coeffs": [-3.0, 2.0]},
            "language": "python",
        }
        task = Task(
            task_id="test_18",
            task_name="test",
            task_type="puzzle",
            level=1,
            prompt="Test",
            expected_answer=expected_output,
            language="python",
            model_output=model_output,
        )
        score = self.reward_fn.compute_reward(task)
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
        task = Task(
            task_id="test_19",
            task_name="test",
            task_type="puzzle",
            level=1,
            prompt="Test",
            expected_answer=expected_output,
            language="python",
            model_output=model_output,
        )
        score = self.reward_fn.compute_reward(task)
        self.assertEqual(score.score, 0.0)

    def test_valid_python_float_decimal(self):
        model_output = """<answer>
```python
def sol(v, d):
    return 0.5
```
</answer>"""
        expected_output = {
            "puzzle": "FloatWithDecimalValue",
            "inputs": {"v": 5, "d": 0.1},
            "language": "python",
        }
        task = Task(
            task_id="test_20",
            task_name="test",
            task_type="puzzle",
            level=1,
            prompt="Test",
            expected_answer=expected_output,
            language="python",
            model_output=model_output,
        )
        score = self.reward_fn.compute_reward(task)
        self.assertEqual(score.score, 1.0)


if __name__ == "__main__":
    unittest.main()
