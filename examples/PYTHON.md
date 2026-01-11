## Instruction

Write a Python program that solves the given problem. Your solution should be complete and executable Python code. Your program should output the result in JSON format. Provide your answer as follows:

```python
[your code here]
```

Example output format: `{"result": [2, 4, 6, 8]}`

## Prompt

Write a Python function that filters a list to keep only even numbers.

## Answer

```json
{"result": [2, 4, 6, 8]}
```

## Response

```python
def filter_even(numbers):
    return [n for n in numbers if n % 2 == 0]

result = filter_even([1, 2, 3, 4, 5, 6, 7, 8])
import json
print(json.dumps({"result": result}))
```

## Reward Function

```python
def reward_fn(model_output, expected_output):
    import re
    import subprocess
    import tempfile
    import os
    import json
    
    # 1. Format Objective (Part A): Extract Python code from markdown
    code_pattern = r"```(?:python)?\s*(.*?)```"
    match = re.search(code_pattern, model_output, re.DOTALL)
    
    if not match:
        return (0.0, 0.0)
    
    code = match.group(1).strip()
    code_format_score = 0.5  # Code block found
    
    # 2. Format Objective (Part B): Validate JSON output format
    try:
        with tempfile.TemporaryDirectory() as tmpdir:
            py_file = os.path.join(tmpdir, 'solution.py')
            with open(py_file, 'w') as f:
                f.write(code)
            
            # Run
            result = subprocess.run(
                ['python', py_file],
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
                expected_json = json.loads(reference_answer.strip())
                correctness_score = 1.0 if actual_json == expected_json else 0.0
            except json.JSONDecodeError:
                json_format_score = 0.0  # Invalid JSON
                correctness_score = 0.0
    except Exception:
        json_format_score = 0.0
        correctness_score = 0.0
    
    format_score = code_format_score + json_format_score
    return (format_score, correctness_score)
```
