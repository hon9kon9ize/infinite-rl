# Infinite RL

This tool generates synthetic RL datasets for LLM preference optimization using the Gemini API. It supports multiple task types including coding, math, summarization, and more, with built-in reward functions for evaluating generated responses.

## Installation

### Option 1: Clone and Install Locally
```bash
git clone https://github.com/hon9kon9ize/infinite-rl.git
cd infinite-rl
pip install .
```

### Option 2: Install Directly from GitHub
```bash
pip install git+https://github.com/hon9kon9ize/infinite-rl.git
```

## Setup

1. Install dependencies and language runtimes:
   
   The installation process will automatically attempt to install required language runtimes:
   - **macOS**: Uses Homebrew to install Node.js, Java 17, g++, Rust, and ts-node
   - **Linux**: Uses apt-get to install Node.js, OpenJDK 17, g++, Rust, and ts-node
   - **Windows**: Provides links for manual installation, ts-node installation via npm if available

2. Set up your Gemini API key in a `.env` file:
   ```
   GOOGLE_API_KEY=your_api_key_here
   ```

## Supported Tasks

### 1. Coding Task
Evaluates LLM-generated code across multiple programming languages with test case validation.

**Supported Languages:**
- Python
- JavaScript
- C++
- Rust
- Java

**Features:**
- Code execution and validation
- Test case evaluation
- Output comparison with similarity scoring
- Detailed error reporting

**Example:**
```python
from infinite_rl import get_reward_functions

reward_fns = get_reward_functions()
coding_fn = reward_fns["coding"]
coding_fn.set_language("python")

# Evaluate with expected output
score, details = coding_fn.compute_reward(
    model_output="print(2 + 2)",
    expected_output="4"
)

# Evaluate with test cases
test_cases = [
    {"input": "1 + 1", "expected_output": "2"},
    {"input": "5 * 3", "expected_output": "15"}
]
score, details = coding_fn.compute_reward(
    model_output="print(eval(input()))",
    test_cases=test_cases
)
```

### 2. Math Task
Evaluates mathematical problem-solving using symbolic computation.

See [examples/MATH.md](examples/MATH.md) for details.

### 3. Summarization Task
Evaluates text summarization quality using semantic similarity.

See [examples/SUMMARIZATION.md](examples/SUMMARIZATION.md) for details.

## Usage

Run the generator using the CLI:

```bash
python main.py --model gemini-3-flash-preview --type coding,math,summarization --num_samples 12 --out output_data
```

### Arguments

- `--model`: The Gemini model name to use (e.g., `gemini-3-flash-preview`, `gemini-3-pro-preview`).
- `--type`: Comma-separated list of task types to generate. Supported: `coding`, `math`, `summarization`.
- `--num_samples`: Total number of samples to generate. They will be distributed evenly across the specified types.
- `--out`: The output directory where `dataset.csv` and `reward_function.py` will be saved.

## Testing

### Testing the RewardExecutor Locally

Test the executor with different programming languages:

```python
from infinite_rl import RewardExecutor

executor = RewardExecutor(timeout=5)

# Test Python
stdout, stderr = executor.run_single("print('Hello, World!')", "python")
print(f"Python: {stdout}")  # Output: Hello, World!

# Test JavaScript
stdout, stderr = executor.run_single("console.log('Hello, World!')", "javascript")
print(f"JavaScript: {stdout}")  # Output: Hello, World!

# Test TypeScript
stdout, stderr = executor.run_single("console.log('Hello, World!')", "typescript")
print(f"TypeScript: {stdout}")  # Output: Hello, World!

# Test C++
code_cpp = """
#include <iostream>
int main() {
    std::cout << "Hello, World!" << std::endl;
    return 0;
}
"""
stdout, stderr = executor.run_single(code_cpp, "cpp")
print(f"C++: {stdout}")
```

### Testing in Google Colab

Install and test in Colab with this notebook:

```python
# Install the package
!pip install git+https://github.com/hon9kon9ize/infinite-rl.git

# Import and test
from infinite_rl import RewardExecutor, get_reward_functions

# Test executor
executor = RewardExecutor(timeout=5)
stdout, stderr = executor.run_single("print(2 + 2)", "python")
print(f"Executor test - Output: {stdout}, Error: {stderr}")

# Test coding reward function
reward_fns = get_reward_functions()
coding_fn = reward_fns["coding"]
coding_fn.set_language("python")

score, details = coding_fn.compute_reward(
    model_output="print(2 + 2)",
    expected_output="4"
)
print(f"Reward Score: {score}")
print(f"Execution Successful: {details['execution_success']}")
print(f"Output Match: {details['output_match']}")
```

## Output

- `dataset.csv`: Contains the generated prompts, chosen responses, and rejected responses with task metadata.
- `reward_function.py`: A Python file containing reward functions for evaluating generated data.

## Project Structure

```
src/
├── executor.py              # Multi-language code executor
├── generator.py             # LLM prompt generation
├── prompts.py               # Task-specific prompts
└── reward_functions/
    ├── reward_function.py   # Base reward function class
    ├── coding.py            # Coding task evaluator
    ├── math.py              # Math task evaluator
    └── summarization.py     # Summarization task evaluator
```

## Architecture

### RewardExecutor
Handles execution of code in multiple languages with timeout protection and error handling. Located in [src/executor.py](src/executor.py).

### Reward Functions
Each task type has a specialized reward function that:
1. Initializes necessary components
2. Executes/evaluates generated content
3. Computes a reward score (0-1)
4. Returns detailed evaluation metrics

All reward functions inherit from `RewardFunction` base class and are accessible via `get_reward_functions()`.
