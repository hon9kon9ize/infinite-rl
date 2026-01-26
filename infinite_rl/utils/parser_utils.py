from typing import Optional


def extract_tag(text: str, tag: Optional[str] = "answer") -> str:
    """Extracts content for a single tag (default: 'answer').

    - `tag` must be a single tag name (str) or None.
    - Returns a single string. If multiple occurrences of the tag are found,
      their contents are joined with newlines.
    """

    if tag is None or tag is "":
        return text

    tag_start = f"<{tag}>"
    tag_end = f"</{tag}>"

    if not text or (tag_start not in text and tag_end not in text):
        return ""

    if tag_end in text:
        text = text.split(tag_end)[0]

    if tag_start in text:
        text = text.split(tag_start)[-1]

    return text
