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
            one_shot_example = f"""
Solve this programming puzzle:

# Sum of Two Numbers

Write a function that returns the sum of two integers.

Write a function that satisfies the following condition:

```python
def sat(result: int, a=5, b=3):
    return result == a + b
```

Your solution should be a python function with this signature:

```python
def sol(a, b):
    pass
```

First, show your reasoning and approach in <{think_tag}> tags (write naturally with proper spaces and punctuation):

<think>
To solve this puzzle, I need to write a function that takes two parameters a and b and returns their sum. The sat function checks if the result equals a + b, so my solution should simply return a + b.
</think>

Then provide your solution in <{answer_tag}> tags. IMPORTANT: Put ONLY the raw code (no markdown backticks) inside the answer tags:

<answer>
def sol(a, b):
    return a + b
</answer>

---

"""
        else:  # javascript
            one_shot_example = f"""
Solve this programming puzzle:

# Sum of Two Numbers

Write a function that returns the sum of two integers.

Write a function that satisfies the following condition:

```javascript
function sat(result, a=5, b=3) {{
    return result === a + b;
}}
```

Your solution should be a javascript function with this signature:

```javascript
function sol(a, b) {{
    // your code here
}}
```

First, show your reasoning and approach in <{think_tag}> tags (write naturally with proper spaces and punctuation):

<think>
To solve this puzzle, I need to write a function that takes two parameters a and b and returns their sum. The sat function checks if the result equals a + b, so my solution should simply return a + b.
</think>

Then provide your solution in <{answer_tag}> tags. IMPORTANT: Put ONLY the raw code (no markdown backticks) inside the answer tags:

<answer>
function sol(a, b) {{
    return a + b;
}}
</answer>

---

"""

    # Language-specific solution template
    if language == "python":
        solution_template = f"""<{answer_tag}>
```python
def sol(...):
    pass
```
</{answer_tag}>"""
    else:  # javascript
        solution_template = f"""<{answer_tag}>
```javascript
function sol(...) {{
  // your code here
}}
```
</{answer_tag}>"""

    prompt = f"""{one_shot_example}

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

{solution_template}"""

    return prompt


def format_truthy_judge_system_prompt(
    question: str,
    chosen: str,
    rejected: str,
    language: str,
) -> str:
    """Format the system prompt for truthy tasks.

    Args:
        question: The original user input prompt
        chosen: The chosen (better) response
        rejected: The rejected (worse) response
        language: The language of the user input (e.g., 'en', 'zh', 'yue')
    Returns:
        Formatted system prompt string for LLM as judge
    """
    system_prompt = f"""You are a helpful and precise assistant for checking the quality of the response. Your task is determining whether the user input is more similar to the "Chosen" response or the "Rejected" response.
    
## Evaluation Criteria
1. **Relevance**: How well does the response address the question?
2. **Language**: Is the response in the same language as the question?
3. **Preference**: Is the response more similar to the "Chosen" example than the "Rejected" example?

## Question:
{question}

## Question Language:
{LANG_MAP.get(language, 'Unknown')}

## Chosen:
{chosen}

## Rejected:
{rejected}
"""
    return system_prompt


def format_truthy_user_prompt(
    system_prompt: str, question: str, think_tag: str, language: str
) -> str:
    """Format the prompt for truth tasks.

    Args:
        question: The original user input prompt
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

**User Input Language:** {LANG_MAP.get(language, 'Unknown')}

**User Input:** {question}
"""
