import unittest
from infinite_rl.reward_functions.coding import JavascriptRewardFunction


class TestJavascriptRewardFunction(unittest.TestCase):
    """Test JavaScript reward function with JavaScript examples."""

    def setUp(self):
        self.reward_fn = JavascriptRewardFunction(task_name="javascript")
        self.reward_fn.initialize()

    def test_valid_js_code_with_json_output(self):
        model_output = """<answer>
```javascript
console.log(JSON.stringify({result: [2, 4, 6, 8]}));
```
</answer>"""
        # Node's JSON.stringify prints compact JSON without spaces
        expected_output = '{"result":[2,4,6,8]}'
        score = self.reward_fn.compute_reward(model_output, expected_output)
        self.assertEqual(score.score, 1.0)

    def test_missing_code_block(self):
        model_output = "<answer>This is just plain text, no code block.</answer>"
        expected_output = '{"result": [2, 4, 6, 8]}'
        score = self.reward_fn.compute_reward(model_output, expected_output)
        self.assertEqual(score.score, 0.0)

    def test_syntax_error_in_code(self):
        # Force a runtime error via typo in console to ensure stderr is produced
        model_output = """<answer>
```javascript
consle.log('x');
```
</answer>"""
        expected_output = "x"
        score = self.reward_fn.compute_reward(model_output, expected_output)

        self.assertEqual(score.score, 0.0)

    def test_order_sensitivity_in_string_matching(self):
        model_output = """<answer>
```javascriptconsole.log('hello world');
```
</answer>"""
        expected_output = "world hello"
        score = self.reward_fn.compute_reward(model_output, expected_output)
        self.assertEqual(score.score, 0.0)

    def test_json_robustness(self):
        model_output1 = """<answer>
```javascript
console.log('{"a": 1, "b": 2}');
```
</answer>"""
        expected_output1 = '{"b": 2, "a": 1}'
        score1 = self.reward_fn.compute_reward(model_output1, expected_output1)
        self.assertEqual(score1.score, 0.0)

    def test_numeric_tolerance(self):
        model_output1 = """<answer>
```javascript
console.log(3.141592653589);
```
</answer>"""
        expected_output1 = 3.141592653590
        score1 = self.reward_fn.compute_reward(model_output1, expected_output1)
        self.assertEqual(score1.score, 0.0)

    def test_whitespace_normalization(self):
        model_output = """<answer>
```javascript
console.log('  hello      world\n\n  ');
```
</answer>"""
        expected_output = "hello world"
        score = self.reward_fn.compute_reward(model_output, expected_output)
        self.assertEqual(score.score, 0.0)

    def test_multiple_code_blocks(self):
        model_output = """<answer>
Some initial thoughts.
```javascript
console.log('first');
```
Some other code.
```javascript
console.log('second');
```
</answer>"""
        score = self.reward_fn.compute_reward(model_output, "first")
        self.assertEqual(score.score, 1.0)

    def test_language_specific_extraction(self):
        model_output = """<answer>
```javascript
console.log('js');
```
```python
print('py')
```
</answer>"""
        score = self.reward_fn.compute_reward(model_output, "js")
        self.assertEqual(score.score, 1.0)

    def test_nested_json_robustness(self):
        nested_data = {"a": [1, {"b": 2}], "c": {"d": [3, 4], "e": "f"}}
        model_output = f"""<answer>
```javascript
console.log(JSON.stringify({nested_data}));
```
</answer>"""
        expected_output = '{"c": {"e": "f", "d": [3, 4]}, "a": [1, {"b": 2}]}'
        score = self.reward_fn.compute_reward(model_output, expected_output)
        self.assertEqual(score.score, 0.0)
