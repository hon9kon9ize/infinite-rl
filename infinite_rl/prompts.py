SYSTEM_PROMPT = """
You are an expert synthetic data generator for Reinforcement Learning from Human Feedback (RLHF). Your goal is to create complex, high-quality training samples.

Each sample MUST follow this structure:

## Prompt
[A challenging, clear instruction or question]

## Answer
[The canonical ground-truth. For technical tasks (code/math), this must be perfectly correct. For code: provide the FULL implementation plus a test block. For math: include <answer>RESULT</answer>.]

## Response
[A helpful, high-quality response. This section MUST include:
1. A step-by-step explanation of the logic.
2. The FINAL solution provided in a markdown code block (e.g., ```python).
3. The code block must be identical or functionally equivalent to the Answer.]

TECHNICAL CONSTRAINTS:
1. NO INTERACTIVE INPUT: Never use `input()`, `readline()`, or any function that waits for user interaction. Use hardcoded test cases.
2. STANDALONE CODE: Code must be completely self-contained (imports included) and runnable.
3. TEST BLOCK: Every code solution (in both Answer and Response) MUST include a markdown code block and MUST end with:
   if __name__ == "__main__":
       # Hardcoded test case
       print(solve_problem(...))
4. MATH WRAPPING: Mathematical answers must always be wrapped in <answer>...</answer> tags.
5. NO EXTERNAL HEADERS: Avoid "## Instruction", "## Reward Function", etc. Only use ## Prompt, ## Answer, ## Response.
6. CODE BLOCKS: ALWAYS wrap code in triple backticks with the language identifier. This is REQUIRED for both Answer and Response.
"""

TYPE_PROMPTS = {
    "python": "Generate a complex Python problem (e.g., algorithms, data structures, or system design). Ensure the solution is fully automated (no input()) and includes a test block.",
    "javascript": "Generate a complex JavaScript/Node.js problem. Use modern ES6+ syntax. Ensure the solution is fully automated and includes a console.log test block.",
    "typescript": "Generate a complex TypeScript problem. Ensure the code is valid TypeScript and includes an execution block.",
    "java": "Generate a Java programming problem. Provide a complete class with a public static void main method for testing.",
    "cpp": "Generate a C++ problem. Provide a complete program with a main function and iostream for test output.",
    "rust": "Generate a Rust problem. Provide a complete runnable program with a main function.",
    "html": "Generate a semantic HTML/CSS layout problem. Ensure the solution is well-formed code.",
    "math": "Generate a complex multi-step mathematical word problem. Reasoning must be step-by-step. Wrap final result in <answer> tags.",
    "summarization": "Generate a text summarization task with a long, detailed document (500+ words). Provide a concise summary in <summary> tags.",
    "coding": "Generate a complex Python algorithm or utility problem. Use an automated test block instead of user input.",
}
