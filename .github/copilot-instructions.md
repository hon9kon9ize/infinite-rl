# Copilot / Agent Instructions for Infinite RL

Purpose: Short, actionable guidance to help AI coding agents be productive in this repository.

## Big picture
- The repo generates synthetic RLHF datasets by orchestrating an LLM (Gemini via the `google.genai` client) to produce task samples and uses *reward functions* to evaluate them.
- **Supported task types**: 
  - **Math** (Level 0): Problem-solving using symbolic computation (SymPy). All math tasks are at level 0, sourced from GSM8K filtered for easy mathematical problems.
  - **Puzzle** (Levels 1-5): Programming challenges in Python (subprocess) and JavaScript (WASM runtime). Difficulty rated 1-5 using Gemini 3 Flash model.
- Key runtime flow:
  1. `scripts/generate.py` -> calls `generate_dataset()` in `infinite_rl/generator.py` to orchestrate sampling.
  2. `generator.py` parses model outputs with `infinite_rl/parser.py`.
  3. Generated samples are evaluated by specialized reward functions in `infinite_rl/reward_functions/`:
     - `MathRewardFunction`: Validates mathematical solutions using symbolic equivalence
     - `PuzzleRewardFunction`: Executes and validates code against puzzle specifications
  4. Puzzle execution: JavaScript via WASM (`infinite_rl/executor.py` + `puzzle_js.wasm`), Python via subprocess (`infinite_rl/runner.py`)
  5. Math evaluation: Uses `sympy` for symbolic computation, references `math.json` for task data (all tasks at level 0)
- Programming puzzles include difficulty ratings (1-5 scale) generated using Gemini 3 Flash model.
- `CurriculumLearning` class (`infinite_rl/curriculum.py`) provides adaptive difficulty progression using **sliding window success rates**:
  - Starts at level 0 (math tasks only)
  - Progresses through levels 1-5 (programming puzzles) based on success rates
  - Tracks last N episodes (default: 50) of success/failure per task type
  - Advances difficulty when success rate > 80% AND variance < 0.05 (configurable)
  - Ensures agent has truly mastered current level, not just "catching up"
  - Per-task-type windows allow independent progression for math and puzzles

## What to know before changing code
- Samples must follow the strict 3-head format: `[PROMPT]`, `[ANSWER]`, `[RESPONSE]` and the final content must be wrapped in `<answer>` tags.
- **Math tasks**: `MathRewardFunction` expects the answer tag to contain a numeric value or symbolic expression that can be parsed and compared symbolically.
- **Puzzle tasks**: `PuzzleRewardFunction` expects the answer tag to contain a code block (triple-backtick) with valid Python or JavaScript code. The code is executed against the puzzle's SAT (satisfaction) function to determine correctness.
- Reward functions return a `RewardFunctionScore(score, info)` with score ranging from 0.0 to 1.0 and `info` holding diagnostic text.
- `generate_dataset()` is idempotent and resumable: it reads/writes `dataset.csv` and appends failures to `failed_dataset.csv` immediately.
- `--task_dist` format is a comma-separated list of floats for available task types (e.g., `0.5,0.5` for 50% math, 50% puzzle).
- Generator has a rectification loop: low-quality outputs are retried and passed through a `RECTIFY_PROMPT` before being retried; be careful modifying retry logic or thresholds.
- Concurrency uses `ThreadPoolExecutor` with explicit locks (`dataset_lock`, `failed_dataset_lock`, `save_lock`)—be precise when adding shared state.
- **Curriculum learning** uses sliding window success rates:
  - `_track_success()` records 1 (success) or 0 (failure) per task type
  - `_update_level()` checks: success_rate > threshold AND variance < variance_threshold for advancement, or success_rate < demote_threshold AND variance < variance_threshold for demotion
  - `get_success_rate()` provides detailed statistics for debugging
  - Thresholds are configurable:
    - `success_rate_threshold` (default: 0.8 = 80%) for advancement
    - `demote_threshold` (default: 0.4 = 40%) for demotion
    - `variance_threshold` (default: 0.05) for stability requirement
  - Per-task-type windows in `self.success_windows` allow independent progression

## Developer workflows & commands
- Always activate the project's virtual environment before running CLI commands (required for consistent environments):
  ```bash
  # Create the venv (if not created)
  python3 -m venv .venv

  # Activate it in your shell
  source .venv/bin/activate
  ```
- NOTE FOR AGENTS: Every time you run shell commands or tests, ensure `.venv/bin/activate` is applied in the session first.

- Tip: install and test with `wasmtime` in your environment when running runtime-dependent tests or executing WASM-based examples.

- Generate dataset locally:
  ```bash
  python scripts/generate.py --num_samples 100 --out_dir ./data --task_dist 0.5,0.5 --threads 4 --debug
  ```
  Note: `--task_dist 0.5,0.5` means 50% math, 50% puzzle. `--debug` saves raw prompts to `data/debug_prompts/` and `failed_dataset.csv` is appended on failure.

- Run unit tests (CI uses unittest discover):
  ```bash
  python -m unittest discover tests
  # Or locally use pytest if preferred: pytest tests/test_reward_functions.py -q
  ```

- Install from git URL (recommended for CI / Colab):
  ```bash
  # Install matching runtimes release (replace runtimes-vX.Y.Z)
  pip install git+https://github.com/owner/repo@runtimes-vX.Y.Z
  # If you run into GitHub API rate limits, set GITHUB_TOKEN first:
  # export GITHUB_TOKEN=...
  ```

- Verify runtimes are present after install:
  ```bash
  python - <<'PY'
  import infinite_rl, os
  runtimes_dir = os.path.join(os.path.dirname(infinite_rl.__file__), 'runtimes')
  print('Runtimes dir:', runtimes_dir)
  print('Contents:', sorted(os.listdir(runtimes_dir)) if os.path.exists(runtimes_dir) else '<missing>')
  PY
  ```
  Expected contents: `math.json`, `puzzle_js.wasm`, `puzzles.json`

- CI specifics: `.github/workflows/ci.yml` installs `nodejs`, `openjdk-17`, `g++` and runs the example suite, then `unittest` discovery.

## Project-specific conventions & patterns
- Strict output format: the parser looks for `[PROMPT]/[ANSWER]/[RESPONSE]` headers and `<answer>` tags; changing parsing requires updating `infinite_rl/parser.py` and tests.
- When adding a new task type:
  - Add a reward function class under `infinite_rl/reward_functions/` and expose it in `get_reward_functions()`.
- Keep `generate_dataset` interfaces stable: many parts (resume logic, distribution, retry/rectify) depend on its signature.

## Integration points & dependencies
- LLM: `google.genai` (Gemini). `GEMINI_API_KEY` must be set for generation to work.
- Code execution: `wasmtime` + packaged WASM runtimes (`puzzle_js.wasm`) in `infinite_rl/runtimes`. The `Executor` exposes `javascript` for puzzles; Python puzzles use local subprocess execution via `infinite_rl/runner.py`.
  - Runtimes are built by the `build_src/build_wasm.sh` script and an automated GitHub Actions workflow (`.github/workflows/build_and_release_runtimes.yml`) uploads them to GitHub Releases.
  - Installation & CI notes:
    - `setup.py` will try to discover the latest release whose tag starts with `runtimes-` and download runtime assets into the package at build time. A `build_py` hook was added so installing from a git URL (e.g. `pip install git+https://github.com/owner/repo@runtimes-vX.Y.Z`) will include WASM files in the built wheel.
    - CI installs from the corresponding runtimes release tag (e.g., `pip install git+https://github.com/owner/repo@runtimes-v0.1.16`) to ensure the wheel bundles the WASM files.
    - If you hit GitHub API rate limits when discovering/downloading assets, set `GITHUB_TOKEN` in the environment before installing.
    - You can override the release tag with `RUNTIME_RELEASE_TAG` or the repo using `RUNTIME_GITHUB_REPO` if needed.

Example usage to pin a release during install:

```bash
# Install from a pinned runtimes release
pip install git+https://github.com/owner/repo@runtimes-v1.2.3

# Or using the environment helper
RUNTIME_RELEASE_TAG=v1.2.3 RUNTIME_GITHUB_REPO=owner/repo python -m pip install .
```
- Math reward: `sympy` is used for symbolic checks. The `math.json` dataset (downloaded during installation) contains math task examples for reference or testing.
- CI installs some language toolchains (Node, Java, g++) even though project currently focuses on Python/JS/TypeScript and math; update CI if you remove language support.

## Quick checks agents should run before PRs
- If changing prompts or task types, update `README.md`, and the `tests/` set accordingly.
- Update CI if you change system dependencies (runtimes, Node/Java/g++ requirements).

## Files to inspect when debugging a change
- `infinite_rl/generator.py` — orchestration, retries/rectify, resume logic, `task_dist` parsing
- `infinite_rl/parser.py` — tag and markdown parsing logic
- `infinite_rl/curriculum.py` — curriculum learning implementation and task difficulty progression
- `infinite_rl/reward_functions/*.py` — reward function implementations and interfaces
- `infinite_rl/executor.py` — how code is run securely (WASM path)
- `infinite_rl/runner.py` — Python puzzle evaluation via local subprocess

---
If something here is unclear or you'd like a different focus (e.g., more examples, a checklist for adding a new task type), tell me what to add and I'll iterate. 👍
- `infinite_rl/curriculum.py` — curriculum learning implementation and task difficulty progression
- `infinite_rl/reward_functions/*.py` — reward function implementations and interfaces
- `infinite_rl/executor.py` — how code is run securely (WASM path)
- `infinite_rl/runner.py` — Python puzzle evaluation via local subprocess

---
If something here is unclear or you'd like a different focus (e.g., more examples, a checklist for adding a new task type), tell me what to add and I'll iterate. 👍