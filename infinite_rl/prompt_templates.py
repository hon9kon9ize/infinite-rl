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
    prompt = f"""Solve this math problem.

**Problem:**
{problem_statement}

**Instructions:**
1. **Reasoning**: You MUST perform your step-by-step reasoning in **English** inside <{think_tag}> tags. Write naturally with proper spaces and punctuation—do NOT remove spaces to compress text.
2. **Final Answer:** Wrap the final numeric value inside <{answer_tag}> tags.

**Response Structure:**
<{think_tag}>
[Reasoning steps in English, with proper spaces and punctuation...]
</{think_tag}>

<{answer_tag}>[Final numeric result]</{answer_tag}>
"""
    return prompt


def format_puzzle_prompt(
    puzzle_data: Dict[str, Any],
    language: str,
    answer_tag: str = "answer",
    think_tag: str = "think",
    one_shot: bool = False,
) -> str:
    """Format a puzzle prompt for the model.

    Always uses answer_tag wrapping with code block inside.

    Args:
        puzzle_data: Dictionary containing puzzle metadata (name, docstring, sat, sol)
        language: Programming language for the puzzle (javascript or python)
        answer_tag: XML tag name for wrapping the answer (default: "answer")
        think_tag: XML tag name for wrapping reasoning (default: "think")
        one_shot: Whether to include a one-shot example (default: False)

    Returns:
        Formatted prompt string with puzzle specification and solution template
    """
    name = puzzle_data.get("name", "")
    docstring = puzzle_data.get("docstring", "")
    sat_func = puzzle_data.get("sat", "")
    sol_func = puzzle_data.get("sol", "")

    # One-shot example section
    one_shot_example = ""
    if one_shot:
        if language == "python":
            one_shot_example = """
**Example Puzzle and Solution:**

# Sum of Two Numbers

Write a function that returns the sum of two integers.

Condition:
```python
def sat(result: int, a=5, b=3):
    return result == a + b
```

Your solution should be:
```python
def sol(a, b):
    pass
```

<think>
To solve this puzzle, I need to write a function that takes two parameters a and b and returns their sum. The sat function checks if the result equals a + b, so my solution should simply return a + b.
</think>

<answer>
```python
def sol(a, b):
    return a + b
```
</answer>

---
"""
        else:  # javascript
            one_shot_example = """
**Example Puzzle and Solution:**

# Sum of Two Numbers

Write a function that returns the sum of two integers.

Condition:
```javascript
function sat(result, a=5, b=3) {
    return result === a + b;
}
```

Your solution should be:
```javascript
function sol(a, b) {
    // your code here
}
```

<think>
To solve this puzzle, I need to write a function that takes two parameters a and b and returns their sum. The sat function checks if the result equals a + b, so my solution should simply return a + b.
</think>

<answer>
```javascript
function sol(a, b) {
    return a + b;
}
```
</answer>

---
"""

    prompt = f"""Solve this programming puzzle:
{one_shot_example}

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

First, show your reasoning and approach in <{think_tag}> tags (write naturally with proper spaces and punctuation):

<{think_tag}>
[Reasoning steps here, must be in English, with proper spacing]
</{think_tag}>

Then provide your solution in <{answer_tag}> tags. IMPORTANT: Put ONLY the raw code (no markdown backticks) inside the answer tags:

<{answer_tag}>
[Your {language} function code here - no triple backticks]
</{answer_tag}>

<{answer_tag}>
```{language}
function sol(...) {{
  // your code here
}}
```
</{answer_tag}>"""

    return prompt


def format_truthy_judge_system_prompt(
    user_input: str,
    chosen: str,
    rejected: str,
) -> str:
    """Format the system prompt for truthy tasks.

    Args:
        user_input: The original user input prompt
        chosen: The chosen (better) response
        rejected: The rejected (worse) response
    Returns:
        Formatted system prompt string for LLM as judge
    """
    system_prompt = f"## User Input:\n{user_input}\n\n## Chosen:\n{chosen}\n\n## Rejected:\n{rejected}"
    return system_prompt


def format_truthy_user_prompt(
    system_prompt: str, user_input: str, think_tag: str
) -> str:
    """Format the prompt for truth tasks.

    Args:
        user_input: The original user input prompt
    Returns:
        Formatted prompt string for truth tasks
    """
    return f"""Analyze the provided System Prompt and User Input. You must first output your internal reasoning process, followed by your final response.

### Constraints
* **Thinking Language:** Your reasoning process MUST be written in **English**, regardless of the language of the user input.
* **Thinking Format:** Wrap this process strictly inside <{think_tag}> tags.
* **Final Answer:** The final response should be in the **same language** as the user input and must directly address the user's request.

## Context
**System Prompt:** {system_prompt}

**User Input:** {user_input}
"""
