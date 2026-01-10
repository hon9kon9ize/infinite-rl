## Instruction

Write a Java program that solves the given problem. Your solution should be a complete, executable Java class. The program should compile and run without errors. Provide your answer as follows:

```java
[your code here]
```

## Question

Write a Java program that takes two integers as input and prints their sum.

## Answer

```java
public class Sum {
    public static void main(String[] args) {
        int a = 5;
        int b = 3;
        int sum = a + b;
        System.out.println(sum);
    }
}
```

## Reward Function

```python
def reward_fn(model_output, reference_answer):
    import re
    import subprocess
    import tempfile
    import os
    
    # 1. Format Objective: Extract Java code and check syntax
    code_pattern = r"```(?:java)?\s*(.*?)```"
    match = re.search(code_pattern, model_output, re.DOTALL)
    
    if not match:
        return (0.0, 0.0)
    
    code = match.group(1).strip()
    format_score = 1.0 if 'class' in code and 'main' in code else 0.0
    
    # 2. Correctness Objective: Check if output matches
    try:
        # Create a temporary file and compile/run
        with tempfile.TemporaryDirectory() as tmpdir:
            java_file = os.path.join(tmpdir, 'Sum.java')
            with open(java_file, 'w') as f:
                f.write(code)
            
            # Compile
            compile_result = subprocess.run(
                ['javac', java_file],
                capture_output=True,
                timeout=5,
                cwd=tmpdir
            )
            
            if compile_result.returncode != 0:
                return (format_score, 0.0)
            
            # Run
            run_result = subprocess.run(
                ['java', '-cp', tmpdir, 'Sum'],
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
