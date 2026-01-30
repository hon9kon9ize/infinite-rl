"""
Prompt formatting templates for task generation.

This module provides functions to format task prompts for both math and puzzle tasks,
ensuring consistent structure across all generated tasks.
"""

from typing import Dict, Any, Optional

# Language code to full language name mapping
LANG_MAP = {
    "zh": "Mandarin",
    "yue": "Cantonese",
    "en": "English",
}


def format_math_prompt(
    problem_statement: str,
    answer_tag: str = "answer",
    language: Optional[str] = None,
    think_tag: str = "think",
) -> str:
    """Format a math problem prompt with explicit answer tag format.

    Math answers should be plain numeric/symbolic values, not code blocks.

    Args:
        problem_statement: The math problem to solve
        answer_tag: XML tag name for wrapping the answer (default: "answer")
        language: Target language for the response (e.g., 'en', 'zh', 'es')
        think_tag: XML tag name for wrapping reasoning (default: "think")

    Returns:
        Formatted prompt string with instructions for the model
    """
    lang_name = LANG_MAP.get(language, language) if language else "English"
    if not lang_name:
        lang_name = "English"
    lang_instruction = f"Your response must be in {lang_name}"

    prompt = f"""Solve this math problem. {lang_instruction}

**Problem:**
{problem_statement}

**Instructions:**
1. **Reasoning**: You MUST perform your step-by-step reasoning in **English** inside <{think_tag}> tags.
2. **Response Language**: All text OUTSIDE of XML tags must be in **{lang_name}**.
3. **Final Answer:** Wrap the final numeric value inside <{answer_tag}> tags.

**Response Structure:**
<{think_tag}>
[Reasoning steps in English...]
</{think_tag}>

[Concluding sentence in {lang_name}]

<{answer_tag}>[Final numeric result]</{answer_tag}>
"""
    return prompt


def format_puzzle_prompt(
    puzzle_data: Dict[str, Any],
    language: str,
    answer_tag: str = "answer",
    think_tag: str = "think",
) -> str:
    """Format a puzzle prompt for the model.

    Always uses answer_tag wrapping with code block inside.

    Args:
        puzzle_data: Dictionary containing puzzle metadata (name, docstring, sat, sol)
        language: Programming language for the puzzle (javascript or python)
        answer_tag: XML tag name for wrapping the answer (default: "answer")
        think_tag: XML tag name for wrapping reasoning (default: "think")

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

```{language}
{sat_func}
```

Your solution should be a {language} function with this signature:

```{language}
{sol_func}
```

First, show your reasoning and approach in <{think_tag}> tags:

<{think_tag}>
[Reasoning steps here, must be in English]
</{think_tag}>

Then provide your solution in <{answer_tag}> tags with a triple-backtick code block:

<{answer_tag}>
```{language}
function sol(...) {{
  // your code here
}}
```
</{answer_tag}>"""

    return prompt


def format_reflective_math_prompt(
    original_prompt: str,
    previous_attempt: str,
    **kwargs,
) -> str:
    """Format a reflective learning prompt for a math task that failed format validation.

    Guides the model to retry a previously failed math task with explicit format requirements.

    Args:
        original_prompt: The original math problem prompt
        previous_attempt: The model's previous attempt (that failed format validation)
        answer_tag: XML tag name for wrapping the answer (default: "answer")
        think_tag: XML tag name for wrapping reasoning (default: "think")

    Returns:
        Formatted reflective prompt with math-specific format guidance
    """
    if not previous_attempt:
        previous_attempt = "<no output recorded>"

    reflective_prompt = f"""You previously attempted this problem, but your response had formatting issues. Review your attempt and solve the problem again.

**Original Problem:**
{original_prompt}

**Your Previous Attempt:**
{previous_attempt}

**Guidelines:**
Please review the following guidelines carefully when solving the problem again:
1. Pay attention to the format requirements
2. Provide your answer in <answer> tags with just the numeric or symbolic result
3. Ensure your response follows the specified format structure

{original_prompt}
"""
    return reflective_prompt


def format_reflective_puzzle_prompt(
    original_prompt: str,
    previous_attempt: str,
    **kwargs,
) -> str:
    """Format a reflective learning prompt for a puzzle/code task that failed format validation.

    Guides the model to retry a previously failed puzzle with explicit format requirements.

    Args:
        original_prompt: The original puzzle prompt
        previous_attempt: The model's previous attempt (that failed format validation)
        language: Programming language for the puzzle (python or javascript)
        answer_tag: XML tag name for wrapping the answer (default: "answer")
        think_tag: XML tag name for wrapping reasoning (default: "think")

    Returns:
        Formatted reflective prompt with puzzle-specific format guidance
    """
    if not previous_attempt:
        previous_attempt = "<no output recorded>"

    reflective_prompt = f"""You previously attempted this puzzle, but your response had formatting issues. Review your attempt and solve this puzzle again.

**Original Task:**
{original_prompt}

**Your Previous Attempt:**
{previous_attempt}

**Guidelines:**
Please review the following guidelines carefully when solving this puzzle again:
1. Pay attention to the format requirements
2. Provide your code solution in <answer> tags with a code block
3. Ensure your solution follows the specified format structure

{original_prompt}
"""
    return reflective_prompt
