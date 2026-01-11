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

## Usage

You can generate a synthetic dataset using the provided script. The generator is designed to be **idempotent and resumable**—if a `dataset.csv` already exists in the output directory, the script will calculate the delta needed to reach your target `--num_samples` while maintaining the requested task distribution.

```bash
python scripts/generate.py --num_samples 100 --out_dir ./my_dataset --threads 4
```

Arguments:
- `--num_samples`: **Target total** number of samples for the dataset (default: 10).
- `--model_name`: Gemini model to use (default: `gemini-2.0-flash-exp`).
- `--out_dir`: Directory to save the `dataset.csv` (default: `data`).
- `--save_every`: Save progress to CSV every N successful samples (default: 1).
- `--threads`: Number of parallel generation threads (default: 1).
- `--max_retries`: Maximum consecutive failed attempts per task type before stopping (default: 5).
- `--timeout`: Timeout (in seconds) for reward function execution (default: 5).
- `--task_dist`: Task distribution as comma-separated floats `[coding, html, math, summarization]` (default: `0.5,0.1,0.3,0.1`).
- `--debug`: Enable verbose logging and save raw LLM responses to `data/debug_prompts`.

**Example for generating only HTML tasks:**
```bash
python scripts/generate.py --num_samples 10 --task_dist 0,1,0,0 --out_dir ./html_only
```

## Testing & Verification

Infinite RL includes a comprehensive testing suite and verification tools to ensure the generator and reward functions are working correctly.

### Run Unit Tests
Use `pytest` to run the unit tests for reward functions and the parser:

```bash
# Run all tests
python -m pytest tests -v

# Run specific reward function tests
python -m pytest tests/test_reward_functions.py -v
```

### Run Example Suite
You can also run the built-in examples to verify that all task types are correctly parsed and evaluated:

```bash
python -m infinite_rl.run_examples
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

# Initialize with custom timeout
reward_fns = get_reward_functions(timeout=10)
coding_fn = reward_fns["coding"]
coding_fn.set_language("python")

# Evaluate with expected output
result = coding_fn.compute_reward(
    model_output="```python\nprint(2 + 2)\n```",
    expected_output="4"
)
print(f"Score: {result.correctness_score}")
```

### 2. Math Task
Evaluates mathematical problem-solving using symbolic computation.

**Example:**
```python
from infinite_rl import get_reward_functions

reward_fns = get_reward_functions()
math_fn = reward_fns["math"]

result = math_fn.compute_reward(
    model_output="<answer>x^2 + 2x + 1</answer>",
    expected_output="(x+1)^2"
)
print(f"Correctness: {result.correctness_score}") 
```

### 3. Summarization Task
Evaluates text summarization quality using semantic similarity.

**Example:**
```python
from infinite_rl import get_reward_functions

reward_fns = get_reward_functions()
summ_fn = reward_fns["summarization"]
# Optional: initialize to load embeddings model early
summ_fn.initialize()

result = summ_fn.compute_reward(
    model_output="<summary>The quick brown fox jumps over the lazy dog.</summary>",
    expected_output="A fast fox leaps over a sleepy canine.",
    original_document="The quick brown fox jumps over the lazy dog. This is a longer text."
)
print(f"Semantic Similarity: {result.correctness_score}")
```

### 4. HTML Task
Evaluates LLM-generated HTML code using syntax validation and CSS selector matching.

**Features:**
- HTML syntax validation using BeautifulSoup
- CSS selector-based element matching
- Support for multiple selectors validation
- Custom validator functions

**Example:**
```python
from infinite_rl import get_reward_functions

reward_fns = get_reward_functions()
html_fn = reward_fns["html"]

# Example 1: Single CSS selector
result = html_fn.compute_reward(
    model_output="<div class='container'><h1>Hello</h1></div>",
    reference_answer="div.container h1"
)
print(f"Correctness: {result.correctness_score}")

# Example 2: Multiple selectors
result = html_fn.compute_reward(
    model_output="<html><body><p id='intro'>Welcome</p></body></html>",
    reference_answer={
        "selectors": ["html", "body", "p#intro"]
    }
)

# Example 3: Custom validator
def validate_structure(soup):
    has_body = soup.body is not None
    has_main = soup.select("main")
    return has_body and len(has_main) > 0

result = html_fn.compute_reward(
    model_output="<html><body><main>Content</main></body></html>",
    reference_answer=validate_structure
)
```

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

# Test Rust
code_rust = """
fn main() {
    println!("Hello, World!");
}
"""
stdout, stderr = executor.run_single(code_rust, "rust")
print(f"Rust: {stdout}")

# Test Java
code_java = """
public class Main {
    public static void main(String[] args) {
        System.out.println("Hello, World!");
    }
}
"""
stdout, stderr = executor.run_single(code_java, "java")
print(f"Java: {stdout}")
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
reward_fns = get_reward_functions(timeout=5)
coding_fn = reward_fns["coding"]
coding_fn.set_language("python")

result = coding_fn.compute_reward(
    model_output="```python\nprint(2 + 2)\n```",
    expected_output="4"
)
print(f"Reward Result: {result}")
print(f"Format Score: {result.format_score}")
print(f"Correctness Score: {result.correctness_score}")
```

> **Tip for TypeScript in Colab**: If you encounter `SyntaxError: Unexpected token '?'`, it means your Node.js version is too old. Run this cell to upgrade Node.js and then **Restart your session**:
> ```bash
> !curl -fsSL https://deb.nodesource.com/setup_20.x | sudo -E bash -
> !sudo apt-get install -y nodejs
> !npm install -g ts-node typescript
> ```

## Project Structure

```
infinite_rl/
├── executor.py              # Multi-language code executor
├── generator.py             # LLM orchestration and resume logic
├── parser.py                # Robust tag extraction and markdown parsing
├── prompts.py               # Task-specific system instructions
└── reward_functions/
    ├── reward_function.py   # Base reward function class
    ├── coding.py            # Coding (Python, JS, C++, etc.) evaluator
    ├── html.py              # HTML/CSS layout evaluator
    ├── math.py              # Symbolic Math evaluator
    └── summarization.py     # Semantic similarity evaluator (Jina-v3)
```

## Output

- `data/dataset.csv`: The primary output containing successful samples (Prompt, Answer, Response, Scores).
- `data/failed_dataset.csv`: Detailed log of failed attempts and rectification errors for troubleshooting.
- `data/debug_prompts/`: Raw system and user prompts sent to the LLM (enabled via `--debug`).

## Architecture

### Standardized Format
All task types are designed for RLHF (Reinforcement Learning from Human Feedback) readiness. Every sample follows a strict three-headed structure:
1.  **Prompt**: The instruction.
2.  **Answer**: The ground-truth reference.
3.  **Response**: A detailed step-by-step reasoning (Chain-of-Thought) where the final solution is **always** wrapped in `<answer>` tags.

### Robust Extraction & Validation
We use a specialized `ExampleParser` with fuzzy logic to extract answers even when the LLM slightly deviates from markdown standards (e.g., malformed tags or missing headers).

### RewardExecutor
Handles execution of code in multiple languages with timeout protection and error handling. Located in [infinite_rl/executor.py](infinite_rl/executor.py).

### Reward Functions
Each task type has a specialized reward function that:
1. Initializes necessary components (e.g., loading Jina-v3 for summarization)
2. Executes/evaluates generated content extracted from `<answer>` tags.
3. Computes a reward score (0-1) combining format and correctness.
4. Returns detailed evaluation metrics.

All reward functions inherit from `RewardFunction` base class and are accessible via `get_reward_functions()`.
