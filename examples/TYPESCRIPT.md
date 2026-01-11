## Instruction

Write a TypeScript program that solves the given problem. Your solution should be complete and executable TypeScript code. Your program should output the result in JSON format. Provide your answer as follows:

```typescript
[your code here]
```

Example output format: `{"reversed": "olleh"}`

## Prompt

Write a TypeScript function that reverses a string.

## Answer

```json
{"reversed": "olleh"}
```

## Response

```typescript
function reverseString(str: string): string {
    return str.split('').reverse().join('');
}

const result = reverseString('hello');
console.log(result);
```

## Reward Function

```python
def reward_fn(model_output, expected_output):
    import re
    import subprocess
    import tempfile
    import os
    import json
    
    # 1. Format Objective (Part A): Extract TypeScript code
    code_pattern = r"```(?:typescript|ts)?\s*(.*?)```"
    match = re.search(code_pattern, model_output, re.DOTALL)
    
    if not match:
        return (0.0, 0.0)
    
    code = match.group(1).strip()
    code_format_score = 0.5  # Code block found
    
    # 2. Format Objective (Part B): Validate JSON output
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
                return (code_format_score, 0.0)
            
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
