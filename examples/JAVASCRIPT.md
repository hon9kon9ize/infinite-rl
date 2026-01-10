## Instruction

Write a JavaScript program that solves the given problem. Your solution should be complete and executable Node.js code. Provide your answer as follows:

```javascript
[your code here]
```

## Question

Write a JavaScript function that calculates the factorial of a number.

## Answer

```javascript
function factorial(n) {
    if (n <= 1) return 1;
    return n * factorial(n - 1);
}

console.log(factorial(5));
```

## Reward Function

```python
def reward_fn(model_output, reference_answer):
    import re
    import subprocess
    import tempfile
    import os
    
    # 1. Format Objective: Extract JavaScript code
    code_pattern = r"```(?:javascript|js)?\s*(.*?)```"
    match = re.search(code_pattern, model_output, re.DOTALL)
    
    if not match:
        return (0.0, 0.0)
    
    code = match.group(1).strip()
    format_score = 1.0 if 'function' in code or 'const' in code or 'let' in code else 0.0
    
    # 2. Correctness Objective: Check if output matches
    try:
        with tempfile.TemporaryDirectory() as tmpdir:
            js_file = os.path.join(tmpdir, 'solution.js')
            with open(js_file, 'w') as f:
                f.write(code)
            
            # Run with node
            result = subprocess.run(
                ['node', js_file],
                capture_output=True,
                timeout=5,
                text=True
            )
            
            actual_output = result.stdout.strip()
            expected_output = reference_answer.strip()
            
            correctness_score = 1.0 if actual_output == expected_output else 0.0
    except Exception:
        correctness_score = 0.0
    
    return (format_score, correctness_score)
```
