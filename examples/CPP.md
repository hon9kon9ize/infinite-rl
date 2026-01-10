## Instruction

Write a C++ program that solves the given problem. Your solution should be a complete, compilable C++ program. Provide your answer as follows:

```cpp
[your code here]
```

## Question

Write a C++ program that checks if a number is prime.

## Answer

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
def reward_fn(model_output, reference_answer):
    import re
    import subprocess
    import tempfile
    import os
    
    # 1. Format Objective: Extract C++ code
    code_pattern = r"```(?:cpp|c\+\+)?\s*(.*?)```"
    match = re.search(code_pattern, model_output, re.DOTALL)
    
    if not match:
        return (0.0, 0.0)
    
    code = match.group(1).strip()
    format_score = 1.0 if '#include' in code and 'main' in code else 0.0
    
    # 2. Correctness Objective: Check if output matches
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
