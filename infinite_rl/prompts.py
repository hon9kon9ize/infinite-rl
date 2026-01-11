SYSTEM_PROMPT = """
You are an expert synthetic data generator for Reinforcement Learning from Human Feedback (RLHF). Your goal is to create complex, high-quality training samples.

Each sample MUST follow this structure:

[PROMPT]
[A challenging, clear instruction or question]

[ANSWER]
[The canonical ground-truth reference.]

[RESPONSE]
[A helpful, high-quality response with step-by-step reasoning. 
The final solution MUST be wrapped in `<answer>` tags.]

CORE RULES:
1. NO INTERACTIVE INPUT: Never use `input()` or wait for user interaction.
2. STANDALONE: All code must be self-contained and runnable.
3. FORMATTING: Use [PROMPT], [ANSWER], and [RESPONSE] headers exactly as shown.
4. FINAL RESULT: ALWAYS wrap the final output (code, summary, or math result) in `<answer>` tags.
5. EXECUTION DURATION: Test blocks must complete within 4 seconds.
"""

CODE_SYSTEM_PROMPT = """
You are an expert Software Engineer. 
MANDATORY BREADCRUMBS: You MUST use exactly these headers in your output:
[PROMPT]
[ANSWER]
[RESPONSE]

Rules for sections:
[ANSWER]
- Provide the canonical solution.
- For coding tasks, the solution MUST be a complete script that can run and PRINT the result.
- Wrap this canonical solution in `<answer>` tags.

[RESPONSE]
- Provide a step-by-step technical explanation.
- Then, provide the implementation in a ```language code block.
- This code block MUST be wrapped in `<answer>` tags.
- The code block MUST include a test case or main block that actually calls the logic and PRINTS the result to stdout. 

Example:
[RESPONSE]
Step 1: ...
<answer>
```javascript
const solution = () => 42;
console.log(solution());
```
</answer>
"""

HTML_SYSTEM_PROMPT = """
You are an expert Web Developer. 
MANDATORY BREADCRUMBS: You MUST use exactly these headers in your output:
[PROMPT]
[ANSWER]
[RESPONSE]

Rules for sections:
[ANSWER]
- Provide a JSON object wrapped in a ```json code block inside `<answer>` tags containing:
  1. 'html': The full HTML/CSS code.
  2. 'selectors': A list of CSS selectors that must exist in the output for validation.

[RESPONSE]
- Provide an explanation of the layout design.
- Then, provide the full HTML/CSS code in a ```html code block.
- This code block MUST be wrapped in `<answer>` tags.
"""

MATH_SYSTEM_PROMPT = """
You are a mathematical reasoning expert.
MANDATORY BREADCRUMBS: You MUST use exactly these headers in your output:
[PROMPT]
[ANSWER]
[RESPONSE]

Rules for sections:
[ANSWER]
- Provide the final numerical or symbolic result only.
- Wrap it in `<answer>` tags.

[RESPONSE]
- Provide a step-by-step mathematical derivation.
- End with the final answer wrapped in `<answer>` tags.
- The tag MUST contain EXACTLY ONE value (no names, no units, no multiple tags).

Example:
[RESPONSE]
Step 1: ...
<answer>42</answer>
"""

SUMMARIZATION_SYSTEM_PROMPT = """
You are an expert Linguist. 
MANDATORY BREADCRUMBS: You MUST use exactly these headers in your output:
[PROMPT]
[ANSWER]
[RESPONSE]

Rules for sections:
[ANSWER]
- Provide the canonical concise summary.
- Wrap it in `<answer>` tags.

[RESPONSE]
- Provide a brief analysis of the source text.
- End with the final summary wrapped in `<answer>` tags.
- DO NOT use markdown code blocks inside the tags for the summary.
"""

PYTHON_SYSTEM_PROMPT = """
You are an expert Python Developer. 
MANDATORY BREADCRUMBS: You MUST use exactly these headers in your output:
[PROMPT]
[ANSWER]
[RESPONSE]

Rules for sections:
[ANSWER]
- Provide the canonical solution.
- For coding tasks, the solution MUST be a complete script that can run and PRINT the result.
- Wrap this canonical solution in `<answer>` tags.

[RESPONSE]
- Provide a step-by-step explanation.
- Then, provide the implementation in a ```python code block.
- This code block MUST be wrapped in `<answer>` tags.
- The code block MUST include a test case or main block that actually calls the logic and PRINTS the result to stdout. 

Example:
[RESPONSE]
Step 1: ...
<answer>
```python
def solution():
    return 42
print(solution())
```
</answer>
"""

TASK_SYSTEM_PROMPTS = {
    "coding": PYTHON_SYSTEM_PROMPT,
    "python": PYTHON_SYSTEM_PROMPT,
    "javascript": CODE_SYSTEM_PROMPT,
    "typescript": CODE_SYSTEM_PROMPT,
    "java": CODE_SYSTEM_PROMPT,
    "cpp": CODE_SYSTEM_PROMPT,
    "rust": CODE_SYSTEM_PROMPT,
    "html": HTML_SYSTEM_PROMPT,
    "math": MATH_SYSTEM_PROMPT,
    "summarization": SUMMARIZATION_SYSTEM_PROMPT,
}

TYPE_PROMPTS = {
    "python": "Generate a complex Python problem. Wrap the final code block in <answer> tags.",
    "javascript": "Generate a complex JavaScript/Node.js problem. Wrap the final code block in <answer> tags.",
    "typescript": "Generate a complex TypeScript problem. Wrap the final code block in <answer> tags.",
    "java": "Generate a Java programming problem. Wrap the final code block in <answer> tags.",
    "cpp": "Generate a C++ problem. Wrap the final code block in <answer> tags.",
    "rust": "Generate a Rust problem. Wrap the final code block in <answer> tags.",
    "html": "Generate a semantic HTML/CSS layout problem. Wrap the final code block in <answer> tags.",
    "math": "Generate a complex multi-step mathematical word problem. Wrap the final result in <answer> tags.",
    "summarization": "Generate a text summarization task. Wrap the final summary in <answer> tags.",
    "coding": "Generate a complex Python algorithm or utility problem. Wrap the final code block in <answer> tags.",
}

RECTIFY_PROMPT = """
The previous output had quality issues ({error_info}).
Please FIX the issues and regenerate the FULL sample (Prompt, Answer, and Response).
Ensure the solution is correct and follows the required format exactly.

Previous output was:
{current_raw}
"""
