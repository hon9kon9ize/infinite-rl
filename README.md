# Infinite-RL

Infinite-RL is a reward functions toolbox for LLM Reinforcement Learning. It provides modular reward functions (coding, math, language detection, length and repetition penalties), utilities for evaluating model responses, and optional dataset generation for synthetic RLHF samples via the Gemini API.

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
   - **macOS**: Uses Homebrew to install Node.js and ts-node
   - **Linux**: Uses apt-get to install Node.js and ts-node
   - **Windows**: Provides links for manual installation, ts-node installation via npm if available

2. Set up your Gemini API key:
   ```shell
   export GEMINI_API_KEY=your_api_key_here
   ```

3. (Optional) Activate the Python virtual environment before using the CLI:
   ```bash
   source .ven/bin/activate
   ```

4. Runtimes (WASM)
   - The JS and MicroPython runtimes are built by `build_src/build_wasm.sh`.
   - A GitHub Actions workflow (`.github/workflows/build_and_release_runtimes.yml`) runs the build and uploads `universal_js.wasm` and `micropython.wasm` to a GitHub Release.
   - During installation, `setup.py` will try to download these runtimes automatically from the latest release (or use the `RUNTIME_RELEASE_TAG` environment variable to pin a release). If you prefer to build locally, run `./build_src/build_wasm.sh` and the generated files will be placed in `infinite_rl/runtimes/`.

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
- `--task_dist`: Task distribution as comma-separated floats `[coding, math]` (default: `0.5,0.5`).
- `--debug`: Enable verbose logging and save raw LLM responses to `data/debug_prompts`.

**Example for generating only math tasks:**
```bash
python scripts/generate.py --num_samples 10 --task_dist 0,1 --out_dir ./math_only
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

## Reward Orchestrator 🔧
A convenience utility that loads available reward functions and (optionally) registers auxiliary rewards such as `repetition` and `length`. The orchestrator can compute a main task reward and aggregate auxiliary signals into a single `RewardFunctionScore`.

- Initialization examples:

```python
from infinite_rl import RewardOrchestrator

# Register repetition and length auxiliary rewards
orch = RewardOrchestrator(timeout=10, include_repetition=True, include_length=True)

# See which reward functions are available
print(orch.available())  # e.g. ['math', 'coding', 'lang_consistency', 'reasoning_steps', 'repetition', 'length']
```

- Compute usage:

```python
# Legacy / simple usage: returns a single RewardFunctionScore for the given task
score = orch.compute("<answer>42</answer>", "42", task="math")
print(score.format_score, score.correctness_score)  # main task scores

# Request auxiliary aggregation by providing a language target. When `lang` is given,
# the orchestrator will include registered auxiliary rewards and return an aggregated
# RewardFunctionScore whose `aux_score` is the sum of auxiliary signals.
agg = orch.compute("<answer>Hello world</answer>", "", task="coding", lang="en")
print(agg.aux_score)  # combined aux signals (lang_consistency + length + repetition when registered)
```

- Access individual auxiliaries:

```python
# You can call specific reward functions directly if you need per-reward details
lang_score = orch.get_fn('lang_consistency').compute_reward("<answer>Hello</answer>", 'en')
length_score = orch.get_fn('length').compute_reward("<answer>short</answer>", 2, is_correct=True)
repetition_score = orch.get_fn('repetition').compute_reward("<answer>hi hi hi</answer>", None)
```

Notes:
- `include_repetition` and `include_length` control whether those auxiliary reward functions are registered with the orchestrator.
- The orchestrator preserves the main task's `format_score` and `correctness_score` (accessible on the aggregated return) and uses `aux_score` to surface auxiliary metrics (e.g., repetition penalty, length signal, language consistency).

## Supported Tasks

### 1. Coding Task
Evaluates LLM-generated code across multiple programming languages with test case validation.

**Supported Languages:**
- Python
- JavaScript

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
    model_output="<answer>\n```python\nprint(2 + 2)\n```\n</answer>",
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

### 3. Reasoning Steps (encouragement bonus)
A small encouragement reward that detects explicit chain-of-thought style reasoning placed inside a `<think>...</think>` block. The `ReasoningStepsRewardFunction` looks for common reasoning indicators (e.g., "first", "second", "finally", "therefore") and awards a modest bonus when multiple indicators are present.

**Behavior:**
- If no `<think>` block is found, no bonus is awarded.
- If 1–2 unique indicators appear in the `<think>` block, a small bonus (0.1) is returned.
- If 3+ unique indicators appear, a larger encouragement bonus (0.2) is returned.

**Example:**
```python
from infinite_rl import get_reward_functions

reward_fns = get_reward_functions()
reason_fn = reward_fns["reasoning_steps"]

model_out = "<think>First, we compute the sum. Second, we verify the result. Finally, we present it.</think>"
score = reason_fn.compute_reward(model_out, expected_output=None)
print(f"Reasoning bonus: {score.correctness_score}")
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

# Embedding-based similarity can be tested if you provide an embeddings runtime.

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
    model_output="<answer>\n```python\nprint(2 + 2)\n```\n</answer>",
    expected_output="4"
)
print(f"Reward Result: {result}")
print(f"Format Score: {result.format_score}")
print(f"Correctness Score: {result.correctness_score}")
```


## Development

### Running Unit Tests
To run all unit tests, install development dependencies and use `pytest`:

```bash
pip install -r requirements_dev.txt
pytest
```

## Project Structure

```
infinite_rl/
├── executor.py              # Multi-language code executor
├── generator.py             # LLM orchestration and resume logic
├── parser.py                # Robust tag extraction and markdown parsing
├── prompts.py               # Task-specific system instructions
└── reward_functions/
    ├── reward_function.py   # Base reward function class
    ├── coding.py            # Coding (Python, JS) evaluator
    └── math.py              # Symbolic Math evaluator
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
1. Initializes necessary components (e.g., loading embedding or ML models)
2. Executes/evaluates generated content extracted from `<answer>` tags.
3. Computes a reward score (0-1) combining format and correctness.
4. Returns detailed evaluation metrics.

All reward functions inherit from `RewardFunction` base class and are accessible via `get_reward_functions()`.

#### Cosine Length Reward (length-based regularizer) ✅
A utility to discourage *verbosity* when the answer is correct and to discourage *laziness* (encourage effort) when the answer is incorrect. Instead of a linear penalty, it uses a cosine curve to create a "sweet spot" for response length.

- Purpose: Prevent overly long correct answers and encourage longer attempts for incorrect answers.
- Math (short): For a normalized x in [0,1], the functions used are:
  - Correct answers (decay after target): R = (cos(pi * x) + 1) / 2  (maps 1 -> 0 over range)
  - Incorrect answers (encourage effort): R = (1 - cos(pi * x)) / 2 (maps 0 -> 1 over range)
- Implementation: See `infinite_rl/reward_functions/length.py` — function `cosine_length_reward(length, min_len=1, max_len=1000, target_len=None, correct=True)`.

Usage example (quick):
```python
from infinite_rl.reward_functions.length import cosine_length_reward

length = 350
len_reward = cosine_length_reward(
    length=length,
    min_len=1,
    max_len=1000,
    target_len=200,  # for correct answers, lengths <= 200 get full credit
    correct=True,
)
# Combine with a base correctness score (example):
final_score = base_correctness_score * len_reward
```

Interactive examples (print to inspect behavior):
```python
from infinite_rl.reward_functions.length import cosine_length_reward

print("Short correct:", cosine_length_reward(10, min_len=1, max_len=1000, target_len=20, correct=True))
print("Long correct:", cosine_length_reward(500, min_len=1, max_len=1000, target_len=200, correct=True))
print("Short incorrect (encourage longer):", cosine_length_reward(5, min_len=1, max_len=1000, correct=False))
print("Moderate incorrect (some effort):", cosine_length_reward(150, min_len=1, max_len=1000, correct=False))
```

Notes:
- For `correct=True`, lengths <= `target_len` receive full reward (1.0); beyond that the reward decays smoothly to 0 at `max_len`.
- For `correct=False`, the reward increases smoothly with length to encourage longer reasoning attempts.
- The function clamps `length` to `[min_len, max_len]` and validates bounds.

#### N-gram Repetition Penalty (anti-repetition) ⚠️
We penalize repeated n-grams to discourage degenerate or looping responses. The penalty is a normalized negative value computed as:

```python
from infinite_rl.reward_functions.repetition import ngram_repetition_reward
penalty = ngram_repetition_reward(text, n=3, weight=-0.1)
```

Behavior:
- Uses simple tokenization (lowercasing and punctuation removal) and counts duplicated n-grams.
- Returns a negative penalty (<= 0) proportional to the fraction of duplicated n-grams in the response; 0 if no duplicates.
- `weight` controls the maximum magnitude (default -0.1).

Quick example (inspect behavior):
```python
from infinite_rl.reward_functions.repetition import ngram_repetition_reward

text = "Hello Hello Hello world world world"
penalty = ngram_repetition_reward(text, n=2, weight=-0.1)
print("Repetition penalty (n=2):", penalty)

# Combine with base score
# final_score = max(0.0, base_correctness_score + penalty)
```

Notes:
- Combine this penalty with the base correctness score (e.g., final_score = max(0.0, base_correctness + penalty)).

