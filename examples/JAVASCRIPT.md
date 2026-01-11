## Instruction

Write a JavaScript program that solves the given problem. Your solution should be complete and executable Node.js code. Your program should output the result in JSON format. Provide your answer as follows:

```javascript
[your code here]
```

Example output format: `{"factorial": 120}`

## Prompt

Write a JavaScript function that calculates the factorial of a number.

## Answer

```json
{"factorial": 120}
```

## Response

```javascript
function factorial(n) {
    if (n <= 1) return 1;
    return n * factorial(n - 1);
}

console.log(JSON.stringify({ factorial: factorial(5) }));
```

## Reward Function

```python
def reward_fn(model_output, expected_output):
    import re
    import subprocess
    import tempfile
    import os
    import json
    
    # 1. Format Objective (Part A): Extract JavaScript code
    code_pattern = r"```(?:javascript|js)?\s*(.*?)```"
    match = re.search(code_pattern, model_output, re.DOTALL)
    
    if not match:
        return (0.0, 0.0)
    
    code = match.group(1).strip()
    code_format_score = 0.5  # Code block found
    
    # 2. Format Objective (Part B): Validate JSON output
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
            
            # Try to parse output as JSON
            try:
                actual_json = json.loads(actual_output)
                json_format_score = 0.5  # Valid JSON
                
                # 3. Correctness Objective: Compare JSON structures
                expected_json = json.loads(expected_output.strip())
                correctness_score = 1.0 if actual_json == expected_json else 0.0
            except json.JSONDecodeError:
                json_format_score = 0.0
                correctness_score = 0.0
    except Exception:
        json_format_score = 0.0
        correctness_score = 0.0
    
    format_score = code_format_score + json_format_score
    return (format_score, correctness_score)
```
