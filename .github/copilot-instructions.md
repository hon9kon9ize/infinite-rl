# Copilot / Agent Instructions for Infinite RL

Purpose: Short, actionable guidance to help AI coding agents be productive in this repository.

## Big picture
- The repo generates synthetic RLHF datasets by orchestrating an LLM (Gemini via the `google.genai` client) to produce task samples and uses *reward functions* to evaluate them.
- Key runtime flow:
  1. `scripts/generate.py` -> calls `generate_dataset()` in `infinite_rl/generator.py` to orchestrate sampling.
  2. `generator.py` calls system prompts in `infinite_rl/prompts.py` and parses model outputs with `infinite_rl/parser.py`.
  3. Generated samples are evaluated by reward functions in `infinite_rl/reward_functions/` (currently `coding` and `math`).
  4. Code execution for evaluation uses WASM runtimes via `infinite_rl/executor.py` (uses `wasmtime` and packaged runtimes in `infinite_rl/runtimes`; currently supports only the `javascript` and `python` runtimes).

## What to know before changing code
- Samples must follow the strict 3-head format: `[PROMPT]`, `[ANSWER]`, `[RESPONSE]` and the final content must be wrapped in `<answer>` tags (see `prompts.py`).
- `CodingRewardFunction` expects the answer to contain a code block (triple-backtick) inside `<answer>`; it executes code with `Executor.run_single`.
- Reward functions return a `RewardFunctionScore(format_score, correctness_score, error_msg, aux_score)`—the `aux_score` is used for auxiliary metrics like repetition, length, or language-consistency signals.
- `generate_dataset()` is idempotent and resumable: it reads/writes `dataset.csv` and appends failures to `failed_dataset.csv` immediately.
- `--task_dist` format is a comma-separated list of floats; current parsing expects 2 values: `[coding,math]` (see `generator.py`).
- Generator has a rectification loop: low-quality outputs are retried and passed through a `RECTIFY_PROMPT` before being retried; be careful modifying retry logic or thresholds.
- Concurrency uses `ThreadPoolExecutor` with explicit locks (`dataset_lock`, `failed_dataset_lock`, `save_lock`)—be precise when adding shared state.

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
  Note: `--debug` saves raw prompts to `data/debug_prompts/` and `failed_dataset.csv` is appended on failure.

- Run example suite (same as CI):
  ```bash
  python -m infinite_rl.run_examples
  ```

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

- CI specifics: `.github/workflows/ci.yml` installs `nodejs`, `openjdk-17`, `g++` and runs the example suite, then `unittest` discovery.

## Project-specific conventions & patterns
- Strict output format: the parser looks for `[PROMPT]/[ANSWER]/[RESPONSE]` headers and `<answer>` tags; changing parsing requires updating `infinite_rl/parser.py` and tests.
- When adding a new task type:
  - Add system prompt and a TYPE_PROMPTS entry in `infinite_rl/prompts.py`.
  - Add a reward function class under `infinite_rl/reward_functions/` and expose it in `get_reward_functions()`.
  - Add an example markdown in `infinite_rl/examples/` (used by tests) and unit tests in `tests/`.
- Keep `generate_dataset` interfaces stable: many parts (resume logic, distribution, retry/rectify) depend on its signature.

## Integration points & dependencies
- LLM: `google.genai` (Gemini). `GEMINI_API_KEY` must be set for generation to work.
- Code execution: `wasmtime` + packaged WASM runtimes (`universal_js.wasm`, `micropython.wasm`) in `infinite_rl/runtimes`. The `Executor` exposes only `javascript` and `python` by default; other runtimes (for example, separate embedding runtimes) were removed and must be provided explicitly if you need them.
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
- Math reward: `sympy` is used for symbolic checks.
- CI installs some language toolchains (Node, Java, g++) even though project currently focuses on Python/JS/TypeScript and math; update CI if you remove language support.

## Quick checks agents should run before PRs
- Run unit tests (`unittest` / `pytest`) and `python -m infinite_rl.run_examples`.
- If changing prompts or task types, update `README.md`, `tests/`, and the `examples/` set accordingly.
- Update CI if you change system dependencies (runtimes, Node/Java/g++ requirements).

## Files to inspect when debugging a change
- `infinite_rl/generator.py` — orchestration, retries/rectify, resume logic, `task_dist` parsing
- `infinite_rl/prompts.py` — system prompts and task-specific seed hints
- `infinite_rl/parser.py` — tag and markdown parsing logic
- `infinite_rl/reward_functions/*.py` — reward function implementations and interfaces
- `infinite_rl/executor.py` — how code is run securely (WASM path)
- `tests/test_reward_functions.py` and `tests/README.md` — canonical examples and unit test expectations

Note: Several example files (JAVASCRIPT/PYTHON) were archived to `archive_examples/` to keep `infinite_rl/examples/` focused on the active task types.

---
If something here is unclear or you'd like a different focus (e.g., more examples, a checklist for adding a new task type), tell me what to add and I'll iterate. 👍