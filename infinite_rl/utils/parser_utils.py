from typing import Optional


def extract_tag(text: str, tag: Optional[str] = "answer", exclude: bool = False) -> str:
    """Extracts content for a single tag (default: 'answer').

    Args:
        text: The input text to search
        tag: The tag name to extract. If None or empty string, returns the original text
        exclude: If False (default), returns the content inside the tags.
                If True, returns everything EXCEPT the tag content (including tag markers).

    Returns:
        - When exclude=False: Content inside the tags (with code blocks extracted if present)
        - When exclude=True: Everything outside the tag content
        - If multiple tag occurrences exist with exclude=False, their contents are joined with newlines
        - If multiple tag occurrences exist with exclude=True, the text between them is returned

    Examples:
        >>> text = "abc\\n<answer>xyz</answer>"
        >>> extract_tag(text, "answer", exclude=False)
        'xyz'
        >>> extract_tag(text, "answer", exclude=True)
        'abc'
    """

    if tag is None or tag == "":
        return text

    import re

    tag_start = f"<{tag}>"
    tag_end = f"</{tag}>"

    if exclude:
        # Return everything EXCEPT the tag content (including the tags themselves)
        pattern = f"{re.escape(tag_start)}(.*?){re.escape(tag_end)}"
        # Remove all occurrences of the tag and its content
        result = re.sub(pattern, "", text, flags=re.DOTALL)
        # Clean up multiple consecutive newlines
        result = re.sub(r"\n\n+", "\n", result)
        # Strip leading/trailing whitespace from the entire result and each line
        result = result.strip()
        # Also strip individual lines to handle cases with leading/trailing spaces per line
        lines = result.split("\n")
        lines = [line.strip() for line in lines if line.strip()]
        return "\n".join(lines)
    else:
        # Return the content inside the tags
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
