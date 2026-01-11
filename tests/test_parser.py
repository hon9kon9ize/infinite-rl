import unittest
import tempfile
import os
from infinite_rl.parser import ExampleParser


class TestExampleParser(unittest.TestCase):
    """Unit tests for ExampleParser."""

    def test_parse_standard_text(self):
        """Test parsing standard markdown text with all headers."""
        text = """
## Prompt
What is 2+2?

## Answer
<answer>4</answer>

## Response
The answer to 2+2 is 4.
"""
        parsed = ExampleParser.parse_text(text)
        self.assertEqual(parsed["prompt"], "What is 2+2?")
        self.assertEqual(parsed["answer"], "4")
        self.assertEqual(parsed["response"], "The answer to 2+2 is 4.")

    def test_parse_text_with_code_blocks(self):
        """Test parsing text where the answer is in a JSON code block."""
        text = """
## Prompt
Filter even numbers.

## Answer
```json
{"result": [2, 4]}
```

## Response
```python
print('{"result": [2, 4]}')
```
"""
        parsed = ExampleParser.parse_text(text)
        self.assertEqual(parsed["prompt"], "Filter even numbers.")
        self.assertEqual(parsed["answer"], '{"result": [2, 4]}')
        self.assertEqual(
            parsed["response"], "```python\nprint('{\"result\": [2, 4]}')\n```"
        )

    def test_parse_text_missing_headers(self):
        """Test parsing text with missing headers."""
        text = """
## Prompt
Only a prompt here.
"""
        parsed = ExampleParser.parse_text(text)
        self.assertEqual(parsed["prompt"], "Only a prompt here.")
        self.assertEqual(parsed["answer"], "")
        self.assertEqual(parsed["response"], "")

    def test_parse_text_with_summary_tags(self):
        """Test parsing text with <summary> tags."""
        text = """
## Answer
<summary>This is a summary.</summary>
"""
        parsed = ExampleParser.parse_text(text)
        self.assertEqual(parsed["answer"], "This is a summary.")

    def test_parse_text_fallback_answer(self):
        """Test that the answer falls back to raw text if no tags or code blocks are found."""
        text = """
## Answer
Just a plain string answer.
"""
        parsed = ExampleParser.parse_text(text)
        self.assertEqual(parsed["answer"], "Just a plain string answer.")

    def test_parse_file(self):
        """Test parsing from a file."""
        content = "## Prompt\nTest file prompt\n## Answer\n<answer>Test</answer>"
        with tempfile.NamedTemporaryFile(mode="w", delete=False) as tmp:
            tmp.write(content)
            tmp_path = tmp.name

        try:
            parsed = ExampleParser.parse_file(tmp_path)
            self.assertEqual(parsed["prompt"], "Test file prompt")
            self.assertEqual(parsed["answer"], "Test")
        finally:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)


if __name__ == "__main__":
    unittest.main()
