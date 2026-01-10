## Instruction

Write a Python program that solves the given problem. Your solution should be complete and executable Python code. Provide your answer as follows:

```python
[your code here]
```

## Question

Write a Python function that filters a list to keep only even numbers.

## Answer

```python
def filter_even(numbers):
    return [n for n in numbers if n % 2 == 0]

result = filter_even([1, 2, 3, 4, 5, 6, 7, 8])
print(result)
```

## Reward Function

```python
def reward_fn(model_output, reference_answer):
    import re
    import subprocess
    import tempfile
    import os
    
    # 1. Format Objective: Extract Python code
    code_pattern = r"```(?:python)?\s*(.*?)```"
    match = re.search(code_pattern, model_output, re.DOTALL)
    
    if not match:
        return (0.0, 0.0)
    
    code = match.group(1).strip()
    format_score = 1.0 if ('def ' in code or 'print' in code) else 0.0
    
    # 2. Correctness Objective: Check if output matches
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
            expected_output = reference_answer.strip()
            
            correctness_score = 1.0 if actual_output == expected_output else 0.0
    except Exception:
        correctness_score = 0.0
    
    return (format_score, correctness_score)
```
