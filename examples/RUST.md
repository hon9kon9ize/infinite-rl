## Instruction

Write a Rust program that solves the given problem. Your solution should be a complete, compilable Rust program. Provide your answer as follows:

```rust
[your code here]
```

## Question

Write a Rust function that finds the maximum element in a vector of integers.

## Answer

```rust
fn find_max(nums: Vec<i32>) -> i32 {
    *nums.iter().max().unwrap()
}

fn main() {
    let numbers = vec![3, 1, 4, 1, 5, 9, 2, 6];
    println!("{}", find_max(numbers));
}
```

## Reward Function

```python
def reward_fn(model_output, reference_answer):
    import re
    import subprocess
    import tempfile
    import os
    
    # 1. Format Objective: Extract Rust code
    code_pattern = r"```(?:rust|rs)?\s*(.*?)```"
    match = re.search(code_pattern, model_output, re.DOTALL)
    
    if not match:
        return (0.0, 0.0)
    
    code = match.group(1).strip()
    format_score = 1.0 if 'fn main' in code else 0.0
    
    # 2. Correctness Objective: Check if output matches
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
                return (format_score, 0.0)
            
            # Run
            run_result = subprocess.run(
                [exe_file],
                capture_output=True,
                timeout=5,
                text=True
            )
            
            actual_output = run_result.stdout.strip()
            expected_output = reference_answer.strip()
            
            correctness_score = 1.0 if actual_output == expected_output else 0.0
    except Exception:
        correctness_score = 0.0
    
    return (format_score, correctness_score)
```
