import re
from typing import Optional


def extract_answer_tags(text: str, tag: Optional[str] = "answer") -> str:
    """Extracts content for a single tag (default: 'answer').

    - `tag` must be a single tag name (str) or None.
    - Returns a single string. If multiple occurrences of the tag are found,
      their contents are joined with newlines.
    """
    if not text:
        return ""

    if tag is None:
        tag = "answer"

    tag_esc = re.escape(tag)

    # 1. Standard tags: <tag>...</tag>
    matches = re.findall(
        rf"<(?:{tag_esc})>(.*?)</(?:{tag_esc})>", text, re.DOTALL | re.IGNORECASE
    )
    if matches:
        return "\n".join(m.strip() for m in matches)

    # 2. Only opening tag? Take everything after it
    opening_pattern = rf"(?:(?:```+|\[|<)\s*(?:{tag_esc})\s*(?:>|\]|```+)?)"
    match = re.search(opening_pattern, text, re.IGNORECASE)
    if match:
        content = text[match.end() :].strip()
        content = re.sub(r"```\s*$", "", content).strip()
        return content

    return ""
