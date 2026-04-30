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


def create_reasoning_language_system_prompt(
    reasoning_language: str,
    think_tag: str = "think",
) -> str:
    """Create a system prompt that instructs the model to reason in a specific language.

    This is used when reasoning_template=True and the model auto-injects thinking tags.
    The system prompt ensures the model knows to use the target language for its CoT.

    Args:
        reasoning_language: Language code for reasoning (e.g., 'en', 'yue', 'zh')
        think_tag: XML tag name for reasoning (default: 'think')

    Returns:
        System prompt string instructing reasoning language.
    """
    lang_name = LANG_MAP.get(reasoning_language, reasoning_language)
    return f"You are a helpful assistant. When you think/reason, you MUST write your entire reasoning process in {lang_name}. " \
           f"Your reasoning should be natural and fluent in {lang_name}, with proper spaces and punctuation. " \
           f"Do NOT switch to English or any other language during reasoning."


def format_math_prompt(
    problem_statement: str,
    answer_tag: str = "answer",
    language: Optional[str] = None,
    think_tag: str = "think",
    reasoning_language: Optional[str] = None,
    reasoning_template: bool = False,
) -> str:
    """Format a math problem prompt with explicit answer tag format.

    Math answers should be plain numeric/symbolic values, not code blocks.

    Args:
        problem_statement: The math problem to solve
        answer_tag: XML tag name for wrapping the answer (default: "answer")
        language: Target language for the response (e.g., 'en', 'zh', 'es')
        think_tag: XML tag name for wrapping reasoning (default: "think")
        reasoning_language: Language for reasoning (e.g., 'en', 'yue', 'zh'). Defaults to 'en'.
        reasoning_template: If True, omit think_tag reasoning instructions
            (model already knows the reasoning format, e.g. Qwen3).

    Returns:
        Formatted prompt string with instructions for the model
    """
    if reasoning_template:
        # Reasoning models (e.g. Qwen3) already know to use think tags.
        # Only ask for the answer in answer_tag.
        prompt = f"""Solve this math problem.

**Problem:**
{problem_statement}

**Instructions:**
Wrap the final numeric value inside <{answer_tag}> tags.

**Response Structure:**
<{answer_tag}>[Final numeric result]</{answer_tag}>
"""
    else:
        reasoning_lang_name = LANG_MAP.get(reasoning_language or "en", "English")
        prompt = f"""Solve this math problem.

**Problem:**
{problem_statement}

**Instructions:**
1. **Reasoning**: You MUST perform your step-by-step reasoning in **{reasoning_lang_name}** inside <{think_tag}> tags. Write naturally with proper spaces and punctuation—do NOT remove spaces to compress text.
2. **Final Answer:** Wrap the final numeric value inside <{answer_tag}> tags.

**Response Structure:**
<{think_tag}>
[Reasoning steps in {reasoning_lang_name}, with proper spaces and punctuation...]
</{think_tag}>

<{answer_tag}>[Final numeric result]</{answer_tag}>
"""
    return prompt


def format_puzzle_prompt(
    puzzle_data: Dict[str, Any],
    language: str,
    answer_tag: str = "answer",
    think_tag: str = "think",
    reasoning_language: Optional[str] = None,
    reasoning_template: bool = False,
) -> str:
    """Format a puzzle prompt for the model.

    Always uses answer_tag wrapping with code block inside.

    Args:
        puzzle_data: Dictionary containing puzzle metadata (name, docstring, sat, sol)
        language: Programming language for the puzzle (javascript or python)
        answer_tag: XML tag name for wrapping the answer (default: "answer")
        think_tag: XML tag name for wrapping reasoning (default: "think")
        reasoning_language: Language for reasoning (e.g., 'en', 'yue', 'zh'). Defaults to 'en'.
        reasoning_template: If True, omit think_tag reasoning instructions
            (model already knows the reasoning format, e.g. Qwen3).

    Returns:
        Formatted prompt string with puzzle specification and solution template
    """
    name = puzzle_data.get("name", "")
    docstring = puzzle_data.get("docstring", "")
    sat_func = puzzle_data.get("sat", "")
    sol_func = puzzle_data.get("sol", "")
    docstring = docstring.strip()  # Remove leading/trailing whitespace
    example = puzzle_data.get("example", {})

    # Remove leading/trailing triple quotes from docstring if present
    if docstring.startswith('"""') and docstring.endswith('"""'):
        docstring = docstring[3:-3].strip()

    # Remove leading indents in docstring
    docstring = "\n".join(line.lstrip() for line in docstring.splitlines())

    # Language-specific solution template (no <answer> wrapper - reward extracts from code block directly)
    if language == "python":
        solution_template = f"""```python
{sol_func}
    [Your code here]
```"""
    else:  # javascript
        solution_template = f"""```javascript
{sol_func} {{
    [Your code here]
}}
```"""

    if reasoning_template:
        # Reasoning models (e.g. Qwen3) already know to use think tags.
        prompt = f"""# {name}

{docstring}

Write a function that satisfies the following condition:

```{language}
{sat_func}
```

Provide your solution as a {language} function with this signature:

{solution_template}"""
    else:
        reasoning_lang_name = LANG_MAP.get(reasoning_language or "en", "English")
        prompt = f"""# {name}

{docstring}

Write a function that satisfies the following condition:

```{language}
{sat_func}
```

First, show your reasoning and approach in <{think_tag}> tags (write naturally with proper spaces and punctuation):

<{think_tag}>
[Reasoning steps here, must be in {reasoning_lang_name}, with proper spacing]
</{think_tag}>

Then provide your solution as a {language} function with this signature:

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
    system_prompt: str, question: str, think_tag: str, language: str,
    reasoning_language: Optional[str] = None,
) -> str:
    """Format the prompt for truth tasks.

    Args:
        question: The original user input prompt
    Returns:
        Formatted prompt string for truth tasks
    """
    reasoning_lang_name = LANG_MAP.get(reasoning_language or "en", "English")
    return f"""Analyze the provided System Prompt and User Input. You must first output your internal reasoning process, followed by your final response.

### Constraints
* **Thinking Language:** Your reasoning process MUST be written in **{reasoning_lang_name}**, regardless of the language of the user input.
* **Thinking Format:** Wrap this process strictly inside <{think_tag}> tags.
* **Final Answer:** The final response should be in the **same language** as the user input and must directly address the user's request.

## Context

**System Prompt:** {system_prompt}

**User Input Language:** {LANG_MAP.get(language, 'Unknown')}

**User Input:** {question}
"""
