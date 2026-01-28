from typing import Optional


def extract_tag(text: str, tag: Optional[str] = "answer") -> str:
    """Extracts content for a single tag (default: 'answer').

    - `tag` must be a single tag name (str) or None.
    - Returns a single string. If multiple occurrences of the tag are found,
      their contents are joined with newlines.
    - Automatically removes triple-backtick code blocks and returns the code content.
    """

    if tag is None or tag == "":
        return text

    import re

    tag_start = f"<{tag}>"
    tag_end = f"</{tag}>"

    # Find all occurrences of the tag
    pattern = f"{re.escape(tag_start)}(.*?){re.escape(tag_end)}"
    matches = re.findall(pattern, text, re.DOTALL)

    if matches:
        combined = "\n".join(matches)
        # Remove triple-backtick code blocks, keeping only the code content
        # Pattern: ```language\ncode\n``` -> extract code
        code_pattern = r"```(?:\w+)?\n?([\s\S]*?)\n?```"
        code_matches = re.findall(code_pattern, combined)
        if code_matches:
            # If there are code blocks, extract and return the code content
            return "\n".join(code_matches)
        return combined
    else:
        return ""
