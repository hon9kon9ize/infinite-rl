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
