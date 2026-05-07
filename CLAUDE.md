# CLAUDE.md

Purpose: Short, actionable guidance to help AI coding agents be productive in this repository.

## Big picture
- The repo provides a modular **reward functions toolbox** for LLM Reinforcement Learning and fine-tuning frameworks like Tunix.
- **Supported task types**: 
  - **Math** (Level 0): Problem-solving using symbolic computation. All math tasks are at level 0, sourced from GSM8K filtered for easy mathematical problems.
  - **Puzzle** (Levels 1-5): Programming challenges in Python (subprocess) and JavaScript (WASM runtime). Difficulty rated 1-5 scale.
  - **Pre-Reasoning** (optional, sampling-rate controlled): Chat/SFT-style examples used to train non-empty reasoning on non-math/non-programming tasks. Supports local JSON/JSONL files and Hugging Face datasets with chat-message rows. Primary score comes from LLM Judge.
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
- **Puzzle tasks**: `PuzzleRewardFunction` expects the answer tag to contain a code block (triple-backtick) with valid Python or JavaScript code. Python code is checked against the Python SAT function; JavaScript code is checked against the JavaScript SAT function inside the WASM runtime. Do not validate JavaScript outputs with Python SAT semantics.
- **Pre-reasoning tasks**: Conversation/SFT quality training where the **primary score IS the LLM Judge score** (not binary). Requires `use_llm_judge=True` with `api_host`, `api_port`, and `model_name`. Accepted dataset sources are local `.jsonl`/`.json` files or Hugging Face dataset names via the optional `datasets` package. Accepted row schemas include `messages`, `conversations`, `conversation`, or fallback `prompt`/`question`/`input` plus `reference_answer`/`answer`/`response`/`completion`/`output`/`chosen`. For chat rows, the final assistant turn is removed from the prompt and used as the reference answer for the judge. **lang_consistency** is computed for pre-reasoning tasks and checks language consistency outside `<think>` tags.
- **LLM Judge** serves two roles:
  1. **Primary evaluator for pre-reasoning tasks**: Rates quality on continuous scale (0.0-1.0)
  2. **Auxiliary evaluator for math/puzzle tasks**: Provides quality feedback independent of correctness gates
  - Requires sglang server running Skywork model (V2-Qwen3-4B)
  - Supports configurable score normalization
  - See `docs/LLM_JUDGE_REWARD_FUNCTION.md` for setup instructions
  - Batch evaluation: Deferred until all generations in a batch are accumulated (when `len(task.generations) >= num_generations`)
  - Uses `compute_rewards_batch()` for efficient batch API calls
  - Updated reward scores are recomputed with judge scores included in final combined score
  - **Format gate applies to final training reward**: For pre-reasoning tasks, invalid format, missing `<answer>`, missing/invalid `</think>`, or placeholder reasoning like `<think>blank</think>` gates the final combined reward to zero regardless of judge quality score.
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
- **Simplified Reward API**: Single-call architecture with deferred batch processing:
  - `compute_reward(task_id, model_output)` → `float`: Primary method that computes and returns the combined score
  - Returns primary score immediately for incomplete batches (< num_generations)
  - When batch completes (>= num_generations), defers LLM Judge computation, recomputes combined scores, and returns final combined score
  - No need for separate `get_reward()` calls - rewards are finalized internally at batch completion
  - Generation accumulation, LLM Judge evaluation, curriculum tracking, and logging all happen within `compute_reward()`
- **Dataset Uniqueness**: Critical for preventing GRPO batching errors:
  - Each dataset row must have a unique identifier to prevent task collision
  - Unique IDs use format: `math_{idx}`, `puzzle_{lang}_{name}`, `pre_reasoning_{idx}`
  - Task selection uses full history weighting to ensure diversity across batches
- **Pre-reasoning sampling**:
  - Configure with `pre_reasoning_dataset`, `pre_reasoning_split`, and `pre_reasoning_learning_rate`.
  - `scripts/train.py --pre_reasoning_dataset ...` requires `--use_llm_judge` because there is no local reference-similarity primary reward.
  - If `--pre_reasoning_learning_rate` is omitted, training defaults it to `1.0` when a pre-reasoning dataset is provided, otherwise `0.0`.
  - Use `num_generations=8` for the intended GRPO pre-reasoning objective.
  - Pre-reasoning tasks never advance or demote the math/puzzle curriculum level.
- **Curriculum learning** uses sliding window success rates:
  - `_track_success_group(level, primary_scores)` records GRPO batch-level success: `group_success = 1 if max_primary == 1.0 else 0`
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

- When changing JavaScript puzzle execution or SAT extraction, regenerate and rebuild local runtime assets before testing:
  ```bash
  python build_src/puzzle_prompt.py
  cp assets/puzzles.json infinite_rl/runtimes/puzzles.json
  esbuild build_src/runner.js --bundle --outfile=build_src/bundled_runner.js --format=esm
  build_src/javy build build_src/bundled_runner.js -o infinite_rl/runtimes/puzzle_js.wasm
  python -m unittest tests.test_runner tests.test_puzzle_reward_function tests.test_javascript_puzzle_examples
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
  Expected contents: `math.json`, `puzzle_js.wasm`, `puzzles.json`

- CI specifics: `.github/workflows/ci.yml` installs `nodejs`, `openjdk-17`, `g++` and runs the example suite, then `unittest` discovery.

## Project-specific conventions & patterns
- Strict output format: the parser looks for `<answer>` tags; changing parsing requires updating tests.
- **Prompt generation**: Use `format_puzzle_prompt()`, `format_math_prompt()`, and `format_pre_reasoning_prompt()` from `infinite_rl.prompt_templates` to create prompts.
- **Puzzle data access**: Use `get_puzzle_data()` and `get_available_puzzles()` from `infinite_rl.puzzles` to access puzzle metadata.
- **Auxiliary reward functions**: Additional metrics blended with primary rewards via `CurriculumLearning._initialize_aux_reward_functions()`:
  - `FormatRewardFunction`: Validates `<answer>` and `<think>/</think>` tag structure. Key behaviors:
    - `allow_explanation_between_tags=True` (default when `use_response_content=True`): allows content between `</think>` and `<answer>` — prevents conflict with `response_content` reward
    - Nested tag detection skips placeholder patterns like `<answer>[Final numeric result]</answer>` and `<answer>...</answer>` (model echoing prompt instructions)
    - Only counts `<answer>` tags AFTER `</think>` for "Multiple tags" check — reasoning section echoes are ignored
    - Real nested tags (e.g., `<answer>315</answer>` in CoT) still rejected by `format_think` (0.0)
    - Placeholder reasoning content such as `<think>blank</think>`, `<think>empty</think>`, `<think>none</think>`, and `<think>no reasoning</think>` scores `format_think=0.0` and gates the final combined reward to zero.
  - `LangConsistencyRewardFunction`: For pre-reasoning tasks — checks language consistency outside `<think>` tags
  - `ReasoningStepsRewardFunction`: Rewards structured reasoning with indicators (English: "First", "Then", "Finally", "Therefore"; Cantonese/Chinese: "首先", "然後", "最後", "所以", etc. — 40+ keywords). Minimum score is 0.0 (no longer -1.0 penalty). Empty reasoning returns 0.0.
  - `ResponseContentRewardFunction`: Rewards brief explanations between `</think>` and `<answer>` tags. Sweet-spot curve: 30-500 chars = 1.0, empty = 0.0, verbose >1000 chars decays to 0.4. Default: enabled (`use_response_content=True`).
  - `LLMJudgeRewardFunction`: Remote LLM-based quality scoring via sglang API
  - `LengthRewardFunction`: Rewards appropriate response length
  - Default `aux_weight = 0.5` (blends auxiliary rewards at 50% weight alongside primary rewards)
- **Reasoning template mode**: Some models use a "reasoning" chat template that auto-injects the opening `<think>` tag into the model output. When the model's chat template handles the opening tag, enable `--reasoning-template` (or `reasoning_template=True` in config) so that:
  - `FormatRewardFunction`, `ReasoningStepsRewardFunction`, and `LengthRewardFunction` extract reasoning content as everything before `</think>` (closing tag) rather than requiring both `<think>...` and `</think>` tags
  - The closing `</think>` tag is always required regardless of the flag
  - The base class `extract_think_content()` method handles both modes transparently
  - Without the flag, the standard `<think>...` + `</think>` format is expected
- When adding a new task type:
  - Add a reward function class under `infinite_rl/reward_functions/` and expose it in `get_reward_functions()` for primary tasks.
  - For auxiliary metrics, add to `_initialize_aux_reward_functions()` in curriculum.py and add configuration handling.
  - Do not add a local primary reward for pre-reasoning; its primary score is LLM Judge.

## Integration points & dependencies
- Code execution: `wasmtime` + packaged WASM runtimes (`puzzle_js.wasm`) in `infinite_rl/runtimes`. The `Executor` exposes `javascript` for puzzles; Python puzzles use local subprocess execution via `infinite_rl/runner.py`.
  - JavaScript puzzle correctness is computed by the JS runtime itself: `runner.py` passes the puzzle's JavaScript `sat` source from `runtimes/puzzles.json`, and `build_src/runner.js` returns `{ result, isCorrect }`. If a JavaScript puzzle has JS SAT metadata but the runtime response lacks `isCorrect`, rebuild `puzzle_js.wasm` instead of falling back to Python SAT.
  - `build_src/puzzle_prompt.py` extracts balanced JS `static sat(...) { ... }` methods and includes module-level helper functions used by SAT checks. Keep `assets/puzzles.json`, `infinite_rl/runtimes/puzzles.json`, and `puzzle_js.wasm` in sync when changing JS puzzle generators or runner behavior.
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
- `infinite_rl/curriculum.py` — curriculum learning implementation with single-call `compute_reward()` API, task difficulty progression, batch LLM Judge evaluation, and deferred combined score computation
- `infinite_rl/reward_functions/*.py` — reward function implementations and interfaces
- `infinite_rl/executor.py` — how code is run securely (WASM path)
- `infinite_rl/runner.py` — Python puzzle evaluation via local subprocess
- `scripts/train.py` — GRPO training integration using simplified `compute_reward()` API
  - `--reasoning-template` flag: Enable when model chat template auto-injects `<think>` tag (see reasoning template mode above)
  - `--pre_reasoning_dataset`, `--pre_reasoning_split`, and `--pre_reasoning_learning_rate`: Configure chat/SFT pre-reasoning training. `--pre_reasoning_dataset` requires `--use_llm_judge`.

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
