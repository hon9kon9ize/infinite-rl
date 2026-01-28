# Infinite-RL
![code coverage](https://img.shields.io/badge/code%20coverage-87%25-green)


Infinite-RL is a reward functions toolbox for LLM Reinforcement Learning. It provides modular reward functions for evaluating programming puzzles, mathematical problems, language detection, and auxiliary metrics like length and repetition penalties. The toolbox includes utilities for model response evaluation and optional dataset generation for synthetic RLHF samples via the Gemini API.

The package includes pre-built datasets for math tasks (`math.json`) and programming puzzles (`puzzles.json`), along with WASM runtimes for secure JavaScript execution.

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
   - The JS runtime is built by `build_src/build_wasm.sh`.
   - A GitHub Actions workflow (`.github/workflows/build_and_release_runtimes.yml`) runs the build and uploads `puzzle_js.wasm` to a GitHub Release.
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
- `--task_dist`: Task distribution as comma-separated floats for available task types (default: `0.5,0.5`).
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

# Run puzzle reward function tests
python -m pytest tests/test_puzzle_reward_function.py -v

# Run tests with coverage report
python -m pytest tests --cov=infinite_rl --cov-report=html --cov-report=term-missing

# Or use the convenience script
python tests/run_coverage.py run      # Run tests with coverage
python tests/run_coverage.py view     # Opens HTML report in browser
python tests/run_coverage.py badge    # Update coverage badge in README
python tests/run_coverage.py all      # Run tests and update badge

# View coverage report in browser
open htmlcov/index.html
```

### Code Coverage
The project maintains >80% code coverage. Coverage reports are generated automatically in CI and can be viewed locally:

- **HTML Report**: `htmlcov/index.html` - Interactive coverage report
- **Terminal Report**: Shows missing lines in terminal output
- **XML Report**: `coverage.xml` - For CI integration
- **GitHub Status Checks**: Coverage percentage shown in PR/commit status
- **Auto-updating Badges**: Coverage badges update automatically in README

### CI/CD Features
- **Coverage Enforcement**: Builds fail if coverage drops below 80%
- **Status Checks**: GitHub shows coverage percentage for each commit
- **Badge Updates**: README coverage badges update automatically on main branch
- **Multiple Report Formats**: HTML, XML, and terminal reports generated

## Supported Tasks

### 1. Puzzle Task
Evaluates LLM-generated solutions to programming puzzles across multiple languages with automated verification.

**Supported Languages:**
- Python (executed locally via subprocess)
- JavaScript (executed via WASM runtime)

**Features:**
- Puzzle solution validation using predefined sat functions
- Support for various puzzle types (algebra, basic math, etc.)
- Secure execution environments (WASM for JS, local subprocess for Python)
- Detailed error reporting
- Difficulty ratings: Each programming puzzle has been rated for difficulty (1-5 scale) using Gemini 2.5 Flash model

**Example:**
```python
from infinite_rl import get_reward_functions

# Initialize with custom timeout
reward_fns = get_reward_functions(timeout=10)
puzzle_fn = reward_fns["puzzle"]

# Evaluate Python puzzle solution
result = puzzle_fn.compute_reward(
    model_output="<answer>\n```python\ndef sol(s):\n    return \"19\"\n```\n</answer>",
    expected_output={"puzzle": "SumOfDigits", "inputs": {"s": 10}, "language": "python"}
)
print(f"Score: {result.score}")
```

**Getting Puzzle Prompts:**
You can access the puzzle prompts programmatically to understand what problems are available or to inspect puzzle specifications:

```python
from infinite_rl.puzzles import get_puzzle_prompt, get_available_puzzles

# Get all available JavaScript puzzles
js_puzzles = get_available_puzzles("javascript")
print(f"Available JS puzzles: {len(js_puzzles)}")
print(f"First few: {js_puzzles[:5]}")

# Get all available Python puzzles
py_puzzles = get_available_puzzles("python")
print(f"Available Python puzzles: {len(py_puzzles)}")

# Get a specific puzzle prompt
prompt = get_puzzle_prompt("QuadraticRoot", "javascript")
if prompt:
    print("QuadraticRoot puzzle prompt:")
    print(prompt[:200] + "..." if len(prompt) > 200 else prompt)
else:
    print("Puzzle not found")
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
print(f"Correctness: {result.score}") 
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
print(f"Reasoning bonus: {score.score}")
```





## Curriculum Learning

The `CurriculumLearning` class provides adaptive task difficulty progression based on model performance using a **sliding window success rate** mechanism. It starts with easy tasks (level 1) and gradually increases difficulty to hard tasks (level 5) as the model demonstrates competence.

**Features:**
- **Sliding Window Tracking**: Tracks the last N episodes (default: 50) of success/failure per task type
- **Dual-Criterion Advancement**: Advances difficulty only when **both** conditions are met:
  - Average success rate > 80% (configurable via `success_rate_threshold`)
  - Variance < 0.05 (configurable via `variance_threshold`)
- **Per-Task-Type Windows**: Maintains independent sliding windows for math and puzzle tasks
- **Weighted Selection**: Avoids recently trained tasks to promote variety
- **Multi-Task Support**: Works with math problems and programming puzzles

This ensures the model has truly *mastered* a difficulty level rather than just "catching up" with lucky guesses.

**Configuration:**
```python
from infinite_rl import CurriculumLearning

# Initialize with default settings
cl = CurriculumLearning()

# Or customize thresholds
cl = CurriculumLearning(
    timeout=10,
    answer_tag="answer",
    think_tag="think",
    window_size=50,              # Track last 50 episodes
    success_rate_threshold=0.8,  # Require 80% success rate
    variance_threshold=0.05,     # Require low variance for consistency
)
```

**Example Usage:**
```python
from infinite_rl import CurriculumLearning

cl = CurriculumLearning()

# Get a task appropriate for current skill level
task = cl.get_prompt()
print(f"Task type: {task['task_type']}, Difficulty level: {task['level']}")
print(f"Prompt: {task['prompt']}")

# Evaluate model response and update learning state
model_response = "<answer>4</answer>"
reward = cl.compute_reward(
    task_id=task['task_id'],
    model_output=model_response
)

print(f"Reward: {reward}")

# Check learning progress
stats = cl.get_learning_stats()
print(f"Current level: {stats['current_level']}")
print(f"Success rate: {stats['sliding_window_stats']['mean_success_rate']:.1%}")
print(f"Variance: {stats['sliding_window_stats']['mean_variance']:.4f}")
```

**Learning Progression:**
- Maintains sliding windows of success/failure for each task type
- Calculates mean success rate and variance for recent performance
- Advances to next difficulty level when success rate and consistency thresholds are both met
- Provides detailed statistics to monitor curriculum progression

**Documentation:**
See [SLIDING_WINDOW_CURRICULUM.md](SLIDING_WINDOW_CURRICULUM.md) for comprehensive documentation including configuration guide, example scenarios, and debugging tips. See [QUICK_REFERENCE.md](QUICK_REFERENCE.md) for a quick start guide.

## Testing

### Testing the RewardExecutor Locally

Test the executor with different programming languages:

```python
from infinite_rl import RewardExecutor

executor = RewardExecutor(timeout=5)

# Test JavaScript puzzle execution
stdout, stderr = executor.run_single('{"puzzle": "SumOfDigits", "inputs": {"s": 10}, "code": "function sol(inputs) { return \'19\'; }"}', "javascript")
print(f"JS Result: {stdout}")

# Test Python puzzle execution (via local runner)
# Note: Python puzzles are executed via subprocess in the reward function, not directly through executor
```

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
python_fn = reward_fns["python"]

result = python_fn.compute_reward(
    model_output="<answer>\n```python\nprint(2 + 2)\n```\n</answer>",
    expected_output="4"
)
print(f"Reward Result: {result}")
print(f"Correctness Score: {result.score}")
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
├── curriculum.py            # Curriculum learning with adaptive difficulty
├── executor.py              # Multi-language code executor (WASM for JS)
├── generator.py             # LLM orchestration and resume logic
├── parser.py                # Robust tag extraction and markdown parsing
├── prompts.py               # Task-specific system instructions
├── runner.py                # Python puzzle execution via subprocess
├── puzzles.py               # Puzzle data loading and utilities
├── reward_functions/
│   ├── reward_function.py   # Base reward function class
│   ├── math.py              # Math task evaluator (symbolic computation)
│   ├── puzzle.py            # Puzzle task evaluator (validates code execution)
│   ├── reasoning_steps.py   # Chain-of-thought bonus reward
│   ├── length.py            # Response length regularizer (cosine decay)
│   ├── repetition.py        # N-gram repetition penalty
│   ├── lang_consistency.py  # Language consistency detection
│   └── format.py            # Format validation
└── runtimes/
    ├── math.json            # Math problem dataset with solutions
    ├── puzzles.json         # Programming puzzle specifications
    └── puzzle_js.wasm       # WASM runtime for JavaScript execution
```

### Task Types

**1. Math Tasks**
- **Source**: `infinite_rl/runtimes/math.json`
- **Evaluation**: Symbolic computation with SymPy
- **Reward Function**: `MathRewardFunction`

**2. Puzzle Tasks**
- **Source**: `infinite_rl/runtimes/puzzles.json`
- **Languages**: Python (subprocess execution) and JavaScript (WASM execution)
- **Evaluation**: Code validation against SAT (satisfaction) functions
- **Reward Function**: `PuzzleRewardFunction`
- **Difficulty**: Rated 1-5 per puzzle


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

