# Copilot / Agent Instructions for Infinite RL

Purpose: Short, actionable guidance to help AI coding agents be productive in this repository.

## Big picture
- The repo provides a modular **reward functions toolbox** for LLM Reinforcement Learning and fine-tuning frameworks like Tunix.
- **Supported task types**: 
  - **Math** (Level 0): Problem-solving using symbolic computation. All math tasks are at level 0, sourced from GSM8K filtered for easy mathematical problems.
  - **Puzzle** (Levels 1-5): Programming challenges in Python (subprocess) and JavaScript (WASM runtime). Difficulty rated 1-5 scale.
  - **Truthy** (All Levels, 20% weight): Conversation-based quality evaluation with multilingual support (yue, zh, en). Primary score from LLM Judge (Skywork Reward Model).
- Key components:
  1. **Reward functions** in `infinite_rl/reward_functions/`:
     - `MathRewardFunction`: Validates mathematical solutions using symbolic equivalence
     - `PuzzleRewardFunction`: Executes and validates code against puzzle specifications
     - `LLMJudgeRewardFunction`: Uses remote LLM-based reward model (Skywork Reward V2-Qwen3-4B) via sglang API for continuous quality scoring
  2. **Execution engines**:
     - Puzzle execution: JavaScript via WASM (`infinite_rl/executor.py` + `puzzle_js.wasm`), Python via subprocess (`infinite_rl/runner.py`)
     - Math evaluation: Symbolic validation of mathematical solutions, references `math.json` for task data (all tasks at level 0)
  3. **Curriculum learning** (`infinite_rl/curriculum.py`): Adaptive difficulty progression using **sliding window success rates**:
     - Starts at level 0 (math tasks only)
     - Progresses through levels 1-5 (programming puzzles) based on success rates
     - Tracks last N episodes (default: 50) of success/failure per task type
     - Advances difficulty when success rate > 80% AND variance < 0.05 (configurable)
     - Ensures agent has truly mastered current level, not just "catching up"
     - Per-task-type windows allow independent progression for math and puzzles
     - **Simplified GRPO batch management**: Clean separation between batches with automatic diversity weighting

## What to know before changing code
- Reward functions are designed to integrate with fine-tuning frameworks (e.g., Tunix).
- **Math tasks**: `MathRewardFunction` expects the answer tag to contain a numeric value or symbolic expression that can be parsed and compared symbolically.
- **Puzzle tasks**: `PuzzleRewardFunction` expects the answer tag to contain a code block (triple-backtick) with valid Python or JavaScript code. The code is executed against the puzzle's SAT (satisfaction) function to determine correctness.
- **Truthy tasks**: Conversation-based quality evaluation where the **primary score IS the LLM Judge score** (not binary). System prompt + prompt + chosen/rejected are provided in conversation format. Requires `use_llm_judge=True` with `api_host`, `api_port`, and `model_name`.
- **LLM Judge** serves two roles:
  1. **Primary evaluator for truthy tasks**: Rates quality on continuous scale (0.0-1.0)
  2. **Auxiliary evaluator for math/puzzle tasks**: Provides quality feedback independent of correctness gates
  - Requires sglang server running Skywork model (V2-Qwen3-4B)
  - Supports configurable score normalization
  - See `docs/LLM_JUDGE_REWARD_FUNCTION.md` for setup instructions
  - `get_judge_scores()` automatically computes missing LLM Judge scores before collecting statistics to ensure up-to-date metrics during training
- **GRPO Task Management**: Simplified approach with clean batch separation:
  - Each GRPO batch gets a fresh task instance from `get_prompt()`
  - `DynamicCurriculumDataset` handles within-batch reuse automatically
  - Dataset row weighting ensures diversity across batches
  - No complex active task tracking - removed error-prone `active_tasks` logic
- **GRPO Batch Architecture**: Clean Task → Generation hierarchy with zero redundancy:
  - `Task.generations`: List of all generations for a task (replaces scattered dicts)
  - `Task.add_generation()`: Adds a new generation with output, rewards, and primary score
  - `Task.latest_generation`: Gets the most recent generation
  - `Session.get_batch_data(task_id)`: Retrieves all generation data for analysis
  - `Session.get_batch_stats(task_id)`: Provides comprehensive batch statistics
  - **Simplified Task Management**: Each GRPO batch gets a fresh task instance; within-batch reuse handled by `DynamicCurriculumDataset` caching; dataset row weighting ensures diversity across batches
- **Dataset Uniqueness**: Critical for preventing GRPO batching errors:
  - Each dataset row must have a unique identifier to prevent task collision
  - Unique IDs use format: `math_{idx}`, `puzzle_{lang}_{name}`, `truthy_{idx}`
  - Task selection uses full history weighting to ensure diversity across batches
- **Curriculum learning** uses sliding window success rates:
  - `_track_success()` records 1 (success) or 0 (failure) per task type
  - `_update_level()` checks: success_rate > threshold AND variance < variance_threshold for advancement, or success_rate < demote_threshold AND variance < variance_threshold for demotion
  - `get_success_rate()` provides detailed statistics for debugging
  - Thresholds are configurable:
    - `success_rate_threshold` (default: 0.8 = 80%) for advancement
    - `demote_threshold` (default: 0.4 = 40%) for demotion
    - `variance_threshold` (default: 0.05) for stability requirement
    - `level_change_cooldown` (default: 5) for minimum steps between level changes to prevent rapid fluctuations
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
- Strict output format: the parser looks for `<answer>` tags; changing parsing requires updating tests.
- **Prompt generation**: Use `format_puzzle_prompt()` and `format_math_prompt()` from `infinite_rl.prompt_templates` to create prompts.
- **Puzzle data access**: Use `get_puzzle_data()` and `get_available_puzzles()` from `infinite_rl.puzzles` to access puzzle metadata.
- **Auxiliary reward functions**: Additional metrics like `FormatRewardFunction`, `LangConsistencyRewardFunction`, `ReasoningStepsRewardFunction`, `RepetitionRewardFunction`, `LengthRewardFunction`, and `LLMJudgeRewardFunction` are initialized via `CurriculumLearning._initialize_aux_reward_functions()` and blended with primary rewards.
- When adding a new task type:
  - Add a reward function class under `infinite_rl/reward_functions/` and expose it in `get_reward_functions()` for primary tasks.
  - For auxiliary metrics, add to `_initialize_aux_reward_functions()` in curriculum.py and add configuration handling.

## Integration points & dependencies
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
- Math reward: Symbolic validation is used for checking mathematical correctness. The `math.json` dataset contains math task examples for reference or testing.
- CI installs some language toolchains (Node, Java, g++) even though project currently focuses on Python/JS/TypeScript and math; update CI if you remove language support.

## Quick checks agents should run before PRs
- If changing prompts or task types, update `README.md`, and the `tests/` set accordingly.
- Update CI if you change system dependencies (runtimes, Node/Java/g++ requirements).

## Files to inspect when debugging a change
- `infinite_rl/curriculum.py` — curriculum learning implementation and task difficulty progression (simplified GRPO batch management)
- `infinite_rl/reward_functions/*.py` — reward function implementations and interfaces
- `infinite_rl/executor.py` — how code is run securely (WASM path)
- `infinite_rl/runner.py` — Python puzzle evaluation via local subprocess

## References

**GSM8K Dataset** (Math tasks source):
```bibtex
@article{cobbe2021gsm8k,
  title={Training Verifiers to Solve Math Word Problems},
  author={Cobbe, Karl and Kosaraju, Vineet and Bavarian, Mohammad and Chen, Mark and Jun, Heewoo and Kaiser, Lukasz and Plappert, Matthias and Tworek, Jerry and Hilton, Jacob and Nakano, Reiichiro and Hesse, Christopher and Schulman, John},
  journal={arXiv preprint arXiv:2110.14168},
  year={2021}
}
```

**Programming Puzzles** (Puzzle tasks source):
```bibtex
@inproceedings{
schuster2021programming,
title={Programming Puzzles},
author={Tal Schuster and Ashwin Kalyan and Alex Polozov and Adam Tauman Kalai},
booktitle={Thirty-fifth Conference on Neural Information Processing Systems Datasets and Benchmarks Track},
year={2021},
url={https://arxiv.org/abs/2106.05784}
}
```

**Python Programming Puzzles Repository** (Implementation source):
We borrowed puzzle implementation code from [Microsoft's Python Programming Puzzles](https://github.com/microsoft/PythonProgrammingPuzzles) repository and implemented a JavaScript version for WASM-based execution.

---
If something here is unclear or you'd like a different focus (e.g., more examples, a checklist for adding a new task type), tell me what to add and I'll iterate. 👍