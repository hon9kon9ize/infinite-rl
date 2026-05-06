import unittest
from unittest.mock import patch

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

    def test_javascript_pool_receives_configured_timeout(self):
        class FakePool:
            def __init__(self):
                self.calls = []

            def evaluate(self, data, timeout):
                self.calls.append((data, timeout))
                return {"isCorrect": False}

        fake_pool = FakePool()
        reward_fn = PuzzleRewardFunction(
            task_name="puzzle", language="javascript", timeout=1
        )
        model_output = """<answer>
```javascript
function sol(inputs) { return "18"; }
```
</answer>"""
        expected_output = {
            "puzzle": "SumOfDigits",
            "inputs": {"s": 10},
            "language": "javascript",
        }
        task = Task(
            task_id="test_js_timeout",
            task_name="test",
            task_type="puzzle",
            level=1,
            prompt="Test",
            expected_answer=expected_output,
            language="javascript",
            model_output=model_output,
        )

        with patch(
            "infinite_rl.reward_functions.puzzle._get_pool",
            return_value=fake_pool,
        ):
            score = reward_fn.compute_reward(task)

        self.assertEqual(score.score, 0.0)
        self.assertEqual(fake_pool.calls[0][1], 1)
        self.assertEqual(fake_pool.calls[0][0]["timeout"], 1)

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

    def test_valid_js_quine_uses_javascript_sat(self):
        """Quine differs across languages, so JS answers must use the JS SAT."""
        model_output = """<answer>
```javascript
function sol() {
    return "\\"Q\\"";
}
```
</answer>"""
        expected_output = {
            "puzzle": "Quine",
            "inputs": {},
            "language": "javascript",
        }
        task = Task(
            task_id="test_js_quine",
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

    def test_python_submission_cannot_patch_sat_checker(self):
        model_output = """<answer>
```python
def sol(s):
    import infinite_rl.python_puzzles.generators.basic as basic
    basic.SumOfDigits.sat = staticmethod(lambda x, s=679: True)
    return "0"
```
</answer>"""
        expected_output = {
            "puzzle": "SumOfDigits",
            "inputs": {"s": 10},
            "language": "python",
        }
        task = Task(
            task_id="test_python_patch_sat",
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

        followup_output = """<answer>
```python
def sol(s):
    return "0"
```
</answer>"""
        followup_task = Task(
            task_id="test_python_patch_sat_followup",
            task_name="test",
            task_type="puzzle",
            level=1,
            prompt="Test",
            expected_answer=expected_output,
            language="python",
            model_output=followup_output,
        )
        followup_score = self.reward_fn.compute_reward(followup_task)
        self.assertEqual(followup_score.score, 0.0)

    def test_python_stderr_is_drained(self):
        model_output = """<answer>
```python
def sol(s):
    import sys
    sys.stderr.write("x" * 1000000)
    sys.stderr.flush()
    return "19"
```
</answer>"""
        expected_output = {
            "puzzle": "SumOfDigits",
            "inputs": {"s": 10},
            "language": "python",
        }
        task = Task(
            task_id="test_python_stderr_drained",
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

    def test_python_timeout_respects_configured_value(self):
        reward_fn = PuzzleRewardFunction(
            task_name="puzzle", language="python", timeout=1
        )
        reward_fn.initialize()
        model_output = """<answer>
```python
def sol(s):
    import time
    time.sleep(2)
    return "19"
```
</answer>"""
        expected_output = {
            "puzzle": "SumOfDigits",
            "inputs": {"s": 10},
            "language": "python",
        }
        task = Task(
            task_id="test_python_timeout",
            task_name="test",
            task_type="puzzle",
            level=1,
            prompt="Test",
            expected_answer=expected_output,
            language="python",
            model_output=model_output,
        )

        score = reward_fn.compute_reward(task)
        self.assertEqual(score.score, 0.0)
        self.assertIn("timed out after 1s", score.info)

    def test_no_answer_tags_code_after_think_close(self):
        """Code block without <answer> tags should work if after </think>."""
        model_output = """Let me think about this.
</think>

```python
def sol(coeffs):
    return 1.0
```"""
        expected_output = {
            "puzzle": "QuadraticRoot",
            "inputs": {"coeffs": [1.0, -3.0, 2.0]},
            "language": "python",
        }
        task = Task(
            task_id="test_21",
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

    def test_no_answer_tags_code_anywhere(self):
        """Code block without <answer> tags should work even without </think>."""
        model_output = """Here's my solution:
```python
def sol(coeffs):
    return 1.0
```"""
        expected_output = {
            "puzzle": "QuadraticRoot",
            "inputs": {"coeffs": [1.0, -3.0, 2.0]},
            "language": "python",
        }
        task = Task(
            task_id="test_22",
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
