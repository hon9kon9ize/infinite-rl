"""
Prompt formatting templates for task generation.

This module provides functions to format task prompts for both math and puzzle tasks,
ensuring consistent structure across all generated tasks.
"""

from typing import Dict, Any, Optional


def format_math_prompt(
    problem_statement: str,
    answer_tag: str = "answer",
    language: Optional[str] = None,
) -> str:
    """Format a math problem prompt with explicit answer tag format.

    Math answers should be plain numeric/symbolic values, not code blocks.

    Args:
        problem_statement: The math problem to solve
        answer_tag: HTML tag name for wrapping the answer (default: "answer")
        language: Target language for the response (e.g., 'en', 'zh', 'es')

    Returns:
        Formatted prompt string with instructions for the model
    """
    lang_instruction = ""
    if language and language != "en":
        lang_instruction = f"\nRespond in {language}. "

    prompt = f"""{problem_statement}{lang_instruction}
Provide your final answer in <{answer_tag}> tags with just the numeric or symbolic result (no code blocks):

<{answer_tag}>42</{answer_tag}>

You may show your work before the answer tag."""
    return prompt


def format_puzzle_prompt(
    puzzle_data: Dict[str, Any],
    language: str,
    answer_tag: str = "answer",
) -> str:
    """Format a puzzle prompt for the model.

    Always uses answer_tag wrapping with code block inside.

    Args:
        puzzle_data: Dictionary containing puzzle metadata (name, docstring, sat, sol)
        language: Programming language for the puzzle (javascript or python)
        answer_tag: HTML tag name for wrapping the answer (default: "answer")

    Returns:
        Formatted prompt string with puzzle specification and solution template
    """
    name = puzzle_data.get("name", "")
    docstring = puzzle_data.get("docstring", "")
    sat_func = puzzle_data.get("sat", "")
    sol_func = puzzle_data.get("sol", "")

    prompt = f"""Solve this programming puzzle:

# {name}

{docstring}

Write a function that satisfies the following condition:

```javascript
{sat_func}
```

Your solution should be a {language} function with this signature:

```javascript
{sol_func}
```

Provide your solution in <{answer_tag}> tags with a triple-backtick code block:

<{answer_tag}>
```{language}
function sol(...) {{
  // your code here
}}
```
</{answer_tag}>"""

    return prompt
