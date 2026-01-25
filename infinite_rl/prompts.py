SYNTHESIS_SYSTEM_PROMPT = """
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

PUZZLE_SYSTEM_PROMPT = """
You are an expert programmer solving programming puzzles.
MANDATORY BREADCRUMBS: You MUST use exactly these headers in your output:
[PROMPT]
[ANSWER]
[RESPONSE]

Rules for sections:
[ANSWER]
- Provide the correct answer for the puzzle.
- Wrap it in `<answer>` tags.

[RESPONSE]
- Provide a step-by-step explanation.
- Then, implement a `sol` function that takes the puzzle inputs and returns the solution.
- The `sol` function MUST be wrapped in `<answer>` tags as a code block.
- The code block should define the `sol` function.

Example for QuadraticRoot:
[RESPONSE]
Step 1: Use the quadratic formula...
<answer>
```python
def sol(coeffs):
    a, b, c = coeffs
    return (-b + (b**2 - 4*a*c)**0.5) / (2*a)
```
</answer>
"""

TASK_SYSTEM_PROMPTS = {
    "coding": PYTHON_SYSTEM_PROMPT,
    "python": PYTHON_SYSTEM_PROMPT,
    "javascript": CODE_SYSTEM_PROMPT,
    "math": MATH_SYSTEM_PROMPT,
    "puzzle": PUZZLE_SYSTEM_PROMPT,
}

TYPE_PROMPTS = {
    "python": "Generate a complex Python problem. Wrap the final code block in <answer> tags.",
    "javascript": "Generate a complex JavaScript/Node.js problem. Wrap the final code block in <answer> tags.",
    "language": "Generate a language-consistency task. Provide an example sentence in a specific language or dialect and ask the model to respond in the same language/dialect. In [ANSWER], provide the expected language code (e.g., 'en', 'yue', 'zh-Hant'). Wrap the final answer in <answer> tags.",
    "math": "Generate a complex multi-step mathematical word problem. Wrap the final result in <answer> tags.",
    "coding": "Generate a complex Python algorithm or utility problem. Wrap the final code block in <answer> tags.",
    "puzzle": "Generate a programming puzzle from the available puzzle generators. Specify the puzzle name and inputs. Ask the model to implement a sol function that solves the puzzle. Wrap the final sol function in <answer> tags.",
}

RECTIFY_PROMPT = """
The previous output had quality issues ({error_info}).
Please FIX the issues and regenerate the FULL sample (Prompt, Answer, and Response).
Ensure the solution is correct and follows the required format exactly.

Previous output was:
{current_raw}
"""
