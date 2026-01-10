## Instruction

Write a TypeScript program that solves the given problem. Your solution should be complete and executable TypeScript code. Provide your answer as follows:

```typescript
[your code here]
```

## Question

Write a TypeScript function that reverses a string.

## Answer

```typescript
function reverseString(str: string): string {
    return str.split('').reverse().join('');
}

const result = reverseString('hello');
console.log(result);
```

## Reward Function

```python
def reward_fn(model_output, reference_answer):
    import re
    import subprocess
    import tempfile
    import os
    
    # 1. Format Objective: Extract TypeScript code
    code_pattern = r"```(?:typescript|ts)?\s*(.*?)```"
    match = re.search(code_pattern, model_output, re.DOTALL)
    
    if not match:
        return (0.0, 0.0)
    
    code = match.group(1).strip()
    format_score = 1.0 if 'function' in code and ':' in code else 0.0
    
    # 2. Correctness Objective: Check if output matches
    try:
        with tempfile.TemporaryDirectory() as tmpdir:
            ts_file = os.path.join(tmpdir, 'solution.ts')
            with open(ts_file, 'w') as f:
                f.write(code)
            
            # Try ts-node first, fall back to node
            result = None
            for cmd in [['ts-node', ts_file], ['node', ts_file]]:
                try:
                    result = subprocess.run(
                        cmd,
                        capture_output=True,
                        timeout=5,
                        text=True,
                        cwd=tmpdir
                    )
                    if result.returncode == 0:
                        break
                except:
                    continue
            
            if result is None or result.returncode != 0:
                return (format_score, 0.0)
            
            actual_output = result.stdout.strip()
            expected_output = reference_answer.strip()
            
            correctness_score = 1.0 if actual_output == expected_output else 0.0
    except Exception:
        correctness_score = 0.0
    
    return (format_score, correctness_score)
```
