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
    lang_instruction = ""
    if language and language != "en":
        lang_instruction = f"\nRespond in {language}. "

    prompt = f"""{problem_statement}{lang_instruction}

First, show your reasoning process in <{think_tag}> tags:

<{think_tag}>
Your step-by-step reasoning here...
</{think_tag}>

Then provide your final answer in <{answer_tag}> tags with just the numeric or symbolic result (no code blocks):

<{answer_tag}>42</{answer_tag}>"""
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

```javascript
{sat_func}
```

Your solution should be a {language} function with this signature:

```javascript
{sol_func}
```

First, show your reasoning and approach in <{think_tag}> tags:

<{think_tag}>
Your analysis of the problem and approach here...
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
    answer_tag: str = "answer",
    think_tag: str = "think",
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

    reflective_prompt = f"""You previously attempted this math problem but your response did not follow the required format.

**Your Original Problem:**
{original_prompt}

**Your Previous Attempt:**
{previous_attempt}

**Format Requirements (IMPORTANT):**
First, show your reasoning in <{think_tag}> tags.
Then, wrap your final answer in <{answer_tag}> tags with a numeric or symbolic value (NOT a code block).

**Correct Format Example:**
<{think_tag}>
Let me work through this step by step...
Therefore, the answer is 42.
</{think_tag}>

<{answer_tag}>42</{answer_tag}>

**Guidelines for this retry:**
1. Work through the math problem step by step in <{think_tag}> tags
2. Show your reasoning and calculations
3. Determine the final numeric or symbolic answer
4. Wrap your reasoning in <{think_tag}>...</{think_tag}> tags
5. Wrap ONLY the final answer value in <{answer_tag}>...</{answer_tag}> tags
6. Do NOT include code blocks (```), only the numeric/symbolic result

**Please solve this problem again, paying close attention to the format requirements:**
{original_prompt}
"""
    return reflective_prompt


def format_reflective_puzzle_prompt(
    original_prompt: str,
    previous_attempt: str,
    language: str = "python",
    answer_tag: str = "answer",
    think_tag: str = "think",
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

    # Language-specific code example
    if language.lower() == "javascript":
        code_example = """function sol(param1, param2) {
  // your implementation
  return result;
}"""
    else:  # python or default
        code_example = """def sol(param1, param2):
    # your implementation
    return result"""

    reflective_prompt = f"""You previously attempted this coding puzzle but your response did not follow the required format.

**Your Original Task:**
{original_prompt}

**Your Previous Attempt:**
{previous_attempt}

**Format Requirements (IMPORTANT):**
First, show your reasoning in <{think_tag}> tags.
Then, wrap your code solution in <{answer_tag}> tags with a triple-backtick code block.

**Correct Format Example:**
<{think_tag}>
Your analysis of the problem and approach here...
</{think_tag}>

<{answer_tag}>
```{language}
{code_example}
```
</{answer_tag}>

**Guidelines for this retry:**
1. Analyze the problem and understand the requirements in <{think_tag}> tags
2. Think through your solution approach and explain your reasoning
3. Write the complete, working code
4. Wrap your reasoning in <{think_tag}>...</{think_tag}> tags
5. Wrap your code in <{answer_tag}>...</{answer_tag}> tags
6. Use triple backticks with language specifier: ```{language}
7. Make sure all function parameters match the sat() function signature
8. Do NOT use input() or interactive features

**Please solve this puzzle again, paying close attention to the format requirements:**
{original_prompt}
"""
    return reflective_prompt
