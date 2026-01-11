"""
Unit tests for reward functions using example data.
Tests cover all reward function types with mocked dependencies.
"""

import unittest
import json
import os
from pathlib import Path
from unittest.mock import patch, MagicMock
from infinite_rl.reward_functions.coding import CodingRewardFunction
from infinite_rl.reward_functions.math import MathRewardFunction
from infinite_rl.reward_functions.summarization import SummarizationRewardFunction
from infinite_rl.reward_functions.html import HtmlRewardFunction
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

    @patch("infinite_rl.reward_functions.summarization.SentenceTransformer")
    def test_summarization_example(self, mock_transformer_class):
        example = self.examples.get("SUMMARIZATION")
        self.assertIsNotNone(example)

        # Mock high similarity for the perfect example
        mock_model = MagicMock()
        mock_transformer_class.return_value = mock_model
        mock_sim = MagicMock()
        mock_sim.item.return_value = 1.0
        mock_model.similarity.return_value = mock_sim

        reward_fn = SummarizationRewardFunction(task_name="summarization")
        reward_fn.initialize()

        # Extract the json field from expected_output if it's there
        # For summarization, the parser gets the raw JSON string
        score = reward_fn.compute_reward(
            example["response"], example["answer"], original_document=example["prompt"]
        )
        self.assertEqual(score.format_score, 1.0)
        self.assertEqual(score.correctness_score, 1.0)

    def test_html_example(self):
        example = self.examples.get("HTML")
        self.assertIsNotNone(example)

        reward_fn = HtmlRewardFunction(task_name="html")
        reward_fn.initialize()

        # HTML reward function expects a dict with selectors
        # The parser gets the raw string from Answer block
        expected = json.loads(example["answer"])

        score = reward_fn.compute_reward(example["response"], expected)
        self.assertEqual(score.format_score, 1.0)
        self.assertEqual(score.correctness_score, 1.0)

    def test_rust_example(self):
        example = self.examples.get("RUST")
        self.assertIsNotNone(example)

        reward_fn = CodingRewardFunction(task_name="rust")
        reward_fn.set_language("rust")
        reward_fn.initialize()

        score = reward_fn.compute_reward(example["response"], example["answer"])
        self.assertEqual(score.format_score, 1.0)
        self.assertEqual(score.correctness_score, 1.0)

    def test_java_example(self):
        example = self.examples.get("JAVA")
        self.assertIsNotNone(example)

        reward_fn = CodingRewardFunction(task_name="java")
        reward_fn.set_language("java")
        reward_fn.initialize()

        score = reward_fn.compute_reward(example["response"], example["answer"])
        self.assertEqual(score.format_score, 1.0)
        self.assertEqual(score.correctness_score, 1.0)

    def test_cpp_example(self):
        example = self.examples.get("CPP")
        self.assertIsNotNone(example)

        reward_fn = CodingRewardFunction(task_name="cpp")
        reward_fn.set_language("cpp")
        reward_fn.initialize()

        score = reward_fn.compute_reward(example["response"], example["answer"])
        self.assertEqual(score.format_score, 1.0)
        self.assertEqual(score.correctness_score, 1.0)

    def test_typescript_example(self):
        example = self.examples.get("TYPESCRIPT")
        self.assertIsNotNone(example)

        reward_fn = CodingRewardFunction(task_name="typescript")
        reward_fn.set_language("typescript")
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

    def test_integer_with_wrong_value(self):
        """Test with integer but wrong value."""
        model_output = "<answer>40</answer>"
        expected_output = 42

        score = self.reward_fn.compute_reward(model_output, expected_output)

        self.assertEqual(score.format_score, 1.0)
        # diff = |40-42| = 2, similarity = max(0, 1 - 2/42) = 0.952...
        self.assertAlmostEqual(score.correctness_score, 0.9523809523809523)


class TestHTMLRewardFunction(unittest.TestCase):
    """Test HTML reward function with selector validation."""

    def setUp(self):
        self.reward_fn = HtmlRewardFunction()
        self.reward_fn.initialize()

    def test_valid_html_with_all_selectors(self):
        """Test valid HTML containing all required selectors."""
        # Note: HTML reward function expects dict, not JSON string
        model_output = """<answer><html>
<body>
    <nav>
        <a href="/">Home</a>
        <a href="/about">About</a>
        <a href="/contact">Contact</a>
    </nav>
    <main>
        <h1>Welcome to My Site</h1>
        <p>This is a sample website.</p>
    </main>
    <footer>
        <p>&copy; 2024 My Site. All rights reserved.</p>
    </footer>
</body>
</html></answer>"""
        expected_output = {"selectors": ["nav", "a", "main", "h1", "footer"]}

        score = self.reward_fn.compute_reward(model_output, expected_output)

        self.assertEqual(score.format_score, 1.0)  # Valid HTML
        self.assertEqual(score.correctness_score, 1.0)  # All selectors match

    def test_missing_html_code_block(self):
        """Test when HTML code block is missing (but still valid HTML)."""
        model_output = "<answer><html><body>Content</body></html></answer>"
        expected_output = {"selectors": ["body"]}

        score = self.reward_fn.compute_reward(model_output, expected_output)

        # BeautifulSoup parses raw HTML as valid, so format_score=1.0
        self.assertEqual(score.format_score, 1.0)
        # body selector is present, so correctness_score=1.0
        self.assertEqual(score.correctness_score, 1.0)

    def test_missing_required_selectors(self):
        """Test HTML missing some required selectors."""
        model_output = """<answer><html>
<body>
    <nav>
        <a href="/">Home</a>
    </nav>
    <main>
        <h1>Welcome</h1>
    </main>
</body>
</html></answer>"""
        expected_output = {"selectors": ["nav", "a", "main", "h1", "footer"]}

        score = self.reward_fn.compute_reward(model_output, expected_output)

        self.assertEqual(score.format_score, 1.0)  # Valid HTML
        self.assertEqual(score.correctness_score, 0.0)  # Missing footer selector


class TestSummarizationRewardFunction(unittest.TestCase):
    """Test summarization reward function with mocked SentenceTransformer."""

    @patch("infinite_rl.reward_functions.summarization.SentenceTransformer")
    def test_valid_summary_with_high_similarity(self, mock_transformer_class):
        """Test valid summary with high semantic similarity."""
        # Setup mock
        mock_model_instance = MagicMock()
        mock_transformer_class.return_value = mock_model_instance

        # Mock embeddings and similarity
        mock_embed_1 = MagicMock()
        mock_embed_2 = MagicMock()
        mock_model_instance.encode.side_effect = [mock_embed_1, mock_embed_2]

        # Mock similarity to return high value
        mock_sim = MagicMock()
        mock_sim.item.return_value = 0.95  # High similarity
        mock_model_instance.similarity.return_value = mock_sim

        reward_fn = SummarizationRewardFunction(task_name="summarization")
        reward_fn.initialize()

        model_output = """<answer>
Remote work is shifting economic activity from city centers to suburbs, forcing changes to zoning.
</answer>"""
        expected_output = json.dumps(
            {"summary": "Remote work shifts economic activity from suburbs."}
        )

        score = reward_fn.compute_reward(model_output, expected_output)

        self.assertEqual(score.format_score, 1.0)
        self.assertEqual(
            score.correctness_score, 0.95
        )  # Direct: returns similarity value

    @patch("infinite_rl.reward_functions.summarization.SentenceTransformer")
    def test_valid_summary_with_low_similarity(self, mock_transformer_class):
        """Test valid summary with low semantic similarity."""
        mock_model_instance = MagicMock()
        mock_transformer_class.return_value = mock_model_instance

        mock_embed_1 = MagicMock()
        mock_embed_2 = MagicMock()
        mock_model_instance.encode.side_effect = [mock_embed_1, mock_embed_2]

        mock_sim = MagicMock()
        mock_sim.item.return_value = 0.3  # Low similarity
        mock_model_instance.similarity.return_value = mock_sim

        reward_fn = SummarizationRewardFunction(task_name="summarization")
        reward_fn.initialize()

        model_output = "<answer>The weather is nice today.</answer>"
        expected_output = json.dumps(
            {"summary": "Remote work is shifting economic activity."}
        )

        score = reward_fn.compute_reward(model_output, expected_output)

        self.assertEqual(score.format_score, 1.0)
        self.assertEqual(
            score.correctness_score, 0.3
        )  # Direct: returns similarity value

    @patch("infinite_rl.reward_functions.summarization.SentenceTransformer")
    def test_missing_summary_tag(self, mock_transformer_class):
        """Test when summary tag is missing."""
        mock_model_instance = MagicMock()
        mock_transformer_class.return_value = mock_model_instance

        reward_fn = SummarizationRewardFunction(task_name="summarization")
        reward_fn.initialize()

        model_output = "This is a summary but no tags."
        expected_output = json.dumps({"summary": "Remote work shifts economics."})

        score = reward_fn.compute_reward(model_output, expected_output)

        self.assertEqual(score.format_score, 0.0)
        self.assertEqual(score.correctness_score, 0.0)

    @patch("infinite_rl.reward_functions.summarization.SentenceTransformer")
    def test_summary_length_scoring(self, mock_transformer_class):
        """Test that format_score reflects summary length relative to document."""
        mock_model_instance = MagicMock()
        mock_transformer_class.return_value = mock_model_instance

        # Mock similarity to always be high so we can focus on format_score
        mock_sim = MagicMock()
        mock_sim.item.return_value = 1.0
        mock_model_instance.similarity.return_value = mock_sim

        reward_fn = SummarizationRewardFunction(task_name="summarization")
        reward_fn.initialize()

        original_doc = "A" * 100  # Exactly 100 chars
        expected_output = "Summary"

        # 1. Best case: < 50% length (e.g. 40 chars)
        short_summary = "<answer>" + ("B" * 40) + "</answer>"
        score1 = reward_fn.compute_reward(
            short_summary, expected_output, original_document=original_doc
        )
        # 1.0 (tags) + 1.0 (length) / 2 = 1.0
        self.assertEqual(score1.format_score, 1.0)

        # 2. Mid case: 75% length (e.g. 75 chars)
        # ratio 0.75 -> length_score = 1.0 - (0.75 - 0.5)*2 = 0.5
        # format_score = (1.0 + 0.5) / 2 = 0.75
        mid_summary = "<answer>" + ("B" * 75) + "</answer>"
        score2 = reward_fn.compute_reward(
            mid_summary, expected_output, original_document=original_doc
        )
        self.assertAlmostEqual(score2.format_score, 0.75)

        # 3. Worst case: >= 100% length (e.g. 100 chars)
        # length_score = 0.0
        # format_score = (1.0 + 0.0) / 2 = 0.5
        long_summary = "<answer>" + ("B" * 100) + "</answer>"
        score3 = reward_fn.compute_reward(
            long_summary, expected_output, original_document=original_doc
        )
        self.assertEqual(score3.format_score, 0.5)

    @patch("infinite_rl.reward_functions.summarization.SentenceTransformer")
    def test_summary_with_callable_validator(self, mock_transformer_class):
        """Test summarization with callable validator function."""
        mock_model_instance = MagicMock()
        mock_transformer_class.return_value = mock_model_instance

        reward_fn = SummarizationRewardFunction(task_name="summarization")
        reward_fn.initialize()

        def custom_validator(summary_text):
            # Check if summary is long enough
            return len(summary_text.split()) >= 10

        model_output = "<answer>Remote work is shifting economic activity from city centers to suburbs.</answer>"
        # 12 words, should pass

        score = reward_fn.compute_reward(model_output, custom_validator)

        self.assertEqual(score.format_score, 1.0)
        self.assertEqual(score.correctness_score, 1.0)


if __name__ == "__main__":
    unittest.main()
