## Instruction

Write a Rust program that solves the given problem. Your solution should be a complete, compilable Rust program. Your program should output the result in JSON format. Provide your answer as follows:

```rust
[your code here]
```

Example output format: `{"max": 9}`

## Prompt

Write a Rust function that finds the maximum element in a vector of integers.

## Answer

```json
{"max": 9}
```

## Response

```rust
fn find_max(nums: Vec<i32>) -> i32 {
    *nums.iter().max().unwrap()
}

fn main() {
    let numbers = vec![3, 1, 4, 1, 5, 9, 2, 6];
    println!("{{\"max\": {}}}", find_max(numbers));
}
```

## Reward Function

```python
def reward_fn(model_output, expected_output):
    import re
    import subprocess
    import tempfile
    import os
    import json
    
    # 1. Format Objective (Part A): Extract Rust code
    code_pattern = r"```(?:rust|rs)?\s*(.*?)```"
    match = re.search(code_pattern, model_output, re.DOTALL)
    
    if not match:
        return (0.0, 0.0)
    
    code = match.group(1).strip()
    code_format_score = 0.5 if 'fn main' in code else 0.0
    
    if code_format_score == 0.0:
        return (0.0, 0.0)
    
    # 2. Format Objective (Part B): Validate JSON output
    try:
        with tempfile.TemporaryDirectory() as tmpdir:
            rs_file = os.path.join(tmpdir, 'solution.rs')
            exe_file = os.path.join(tmpdir, 'solution')
            
            with open(rs_file, 'w') as f:
                f.write(code)
            
            # Compile
            compile_result = subprocess.run(
                ['rustc', rs_file, '-o', exe_file],
                capture_output=True,
                timeout=30
            )
            
            if compile_result.returncode != 0:
                return (code_format_score, 0.0)
            
            # Run
            run_result = subprocess.run(
                [exe_file],
                capture_output=True,
                timeout=5,
                text=True
            )
            
            actual_output = run_result.stdout.strip()
            
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
