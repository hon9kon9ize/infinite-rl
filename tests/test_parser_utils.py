import unittest
from infinite_rl.utils.parser_utils import extract_tag


class TestParserUtils(unittest.TestCase):
    """Test parser utility functions."""

    def test_extract_tag_basic(self):
        """Test basic tag extraction."""
        text = "<answer>42</answer>"
        result = extract_tag(text)
        self.assertEqual(result, "42")

    def test_extract_tag_with_content_around(self):
        """Test tag extraction with content before and after."""
        text = "Some text <answer>42</answer> more text"
        result = extract_tag(text)
        self.assertEqual(result, "42")

    def test_extract_tag_custom_tag(self):
        """Test extraction with custom tag name."""
        text = "<result>hello world</result>"
        result = extract_tag(text, "result")
        self.assertEqual(result, "hello world")

    def test_extract_tag_no_tag(self):
        """Test extraction when tag is not present."""
        text = "No tags here"
        result = extract_tag(text)
        self.assertEqual(result, "")

    def test_extract_tag_empty_string(self):
        """Test extraction with empty string."""
        result = extract_tag("")
        self.assertEqual(result, "")

    def test_extract_tag_none_tag(self):
        """Test extraction with None tag (should return original text)."""
        text = "original text"
        result = extract_tag(text, None)
        self.assertEqual(result, "original text")

    def test_extract_tag_empty_tag(self):
        """Test extraction with empty tag string."""
        text = "original text"
        result = extract_tag(text, "")
        self.assertEqual(result, "original text")

    def test_extract_tag_only_start_tag(self):
        """Test extraction with only start tag present."""
        text = "before <answer>content"
        result = extract_tag(text)
        self.assertEqual(result, "")  # Should return empty for incomplete tags

    def test_extract_tag_only_end_tag(self):
        """Test extraction with only end tag present."""
        text = "content</answer> after"
        result = extract_tag(text)
        self.assertEqual(result, "")

    def test_extract_tag_multiple_occurrences(self):
        """Test extraction with multiple tag occurrences."""
        text = "<answer>first</answer> middle <answer>second</answer>"
        result = extract_tag(text)
        # According to docstring, should join with newlines
        self.assertEqual(result, "first\nsecond")

    def test_extract_tag_nested_tags(self):
        """Test extraction with nested tags."""
        text = "<answer>outer <inner>inner content</inner> content</answer>"
        result = extract_tag(text)
        self.assertEqual(result, "outer <inner>inner content</inner> content")

    def test_extract_tag_multiline_content(self):
        """Test extraction with multiline content."""
        text = "<answer>\n  line 1\n  line 2\n</answer>"
        result = extract_tag(text)
        self.assertEqual(result, "\n  line 1\n  line 2\n")

    # Tests for exclude parameter
    def test_extract_tag_exclude_basic(self):
        """Test exclude=True returns content outside the tag."""
        text = "abc\n<answer>xyz</answer>"
        result = extract_tag(text, exclude=True)
        self.assertEqual(result, "abc")

    def test_extract_tag_exclude_with_trailing(self):
        """Test exclude=True with content after tag."""
        text = "before\n<answer>content</answer>\nafter"
        result = extract_tag(text, exclude=True)
        self.assertEqual(result, "before\nafter")

    def test_extract_tag_exclude_no_tag(self):
        """Test exclude=True when tag is not present."""
        text = "no tags here"
        result = extract_tag(text, exclude=True)
        self.assertEqual(result, "no tags here")

    def test_extract_tag_exclude_multiple_tags(self):
        """Test exclude=True with multiple tag occurrences."""
        text = "first<answer>remove1</answer>middle<answer>remove2</answer>last"
        result = extract_tag(text, exclude=True)
        self.assertEqual(result, "firstmiddlelast")

    def test_extract_tag_exclude_empty_outside(self):
        """Test exclude=True with only tag content."""
        text = "<answer>only content</answer>"
        result = extract_tag(text, exclude=True)
        self.assertEqual(result, "")

    def test_extract_tag_exclude_custom_tag(self):
        """Test exclude=True with custom tag name."""
        text = "keep this\n<result>remove this</result>\nand this"
        result = extract_tag(text, "result", exclude=True)
        self.assertEqual(result, "keep this\nand this")

    def test_extract_tag_exclude_whitespace(self):
        """Test exclude=True handles whitespace correctly."""
        text = "  content  \n<answer>remove</answer>\n  more  "
        result = extract_tag(text, exclude=True)
        self.assertEqual(result, "content\nmore")

    def test_extract_tag_exclude_false_explicit(self):
        """Test explicit exclude=False (default behavior)."""
        text = "before <answer>extract</answer> after"
        result = extract_tag(text, exclude=False)
        self.assertEqual(result, "extract")

    def test_extract_tag_code_blocks_with_exclude_false(self):
        """Test code block extraction still works with exclude=False."""
        text = "<answer>\n```python\ndef hello():\n    pass\n```\n</answer>"
        result = extract_tag(text, exclude=False)
        self.assertEqual(result, "def hello():\n    pass")

    def test_extract_tag_exclude_true_preserves_code_in_outside(self):
        """Test exclude=True preserves code blocks in outside content."""
        text = "```python\nprint('keep')\n```\n<answer>remove</answer>"
        result = extract_tag(text, exclude=True)
        self.assertEqual(result, "```python\nprint('keep')\n```")
