## Instruction

Write a C++ program that solves the given problem. Your solution should be a complete, compilable C++ program. Your program should output the result in JSON format. Provide your answer as follows:

```cpp
[your code here]
```

Example output format: `{"is_prime": 1}`

## Prompt

Write a C++ program that checks if a number is prime.

## Answer

```json
{"is_prime": 1}
```

## Response

```cpp
#include <iostream>
using namespace std;

bool isPrime(int n) {
    if (n <= 1) return false;
    if (n <= 3) return true;
    if (n % 2 == 0 || n % 3 == 0) return false;
    for (int i = 5; i * i <= n; i += 6) {
        if (n % i == 0 || n % (i + 2) == 0) return false;
    }
    return true;
}

int main() {
    cout << isPrime(17) << endl;
    return 0;
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
    
    # 1. Format Objective (Part A): Extract C++ code
    code_pattern = r"```(?:cpp|c\+\+)?\s*(.*?)```"
    match = re.search(code_pattern, model_output, re.DOTALL)
    
    if not match:
        return (0.0, 0.0)
    
    code = match.group(1).strip()
    code_format_score = 0.5 if '#include' in code and 'main' in code else 0.0
    
    if code_format_score == 0.0:
        return (0.0, 0.0)
    
    # 2. Format Objective (Part B): Validate JSON output
    try:
        with tempfile.TemporaryDirectory() as tmpdir:
            cpp_file = os.path.join(tmpdir, 'solution.cpp')
            exe_file = os.path.join(tmpdir, 'solution')
            
            with open(cpp_file, 'w') as f:
                f.write(code)
            
            # Compile
            compile_result = subprocess.run(
                ['g++', cpp_file, '-o', exe_file],
                capture_output=True,
                timeout=10
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
