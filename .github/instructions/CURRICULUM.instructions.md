---
applyTo: "infinite_rl/curriculum.py,infinite_rl/reward_functions/**/*.py"
---

# Curriculum Learning & Reward Functions Instructions

## Overview

This project implements a curriculum learning system that progressively increases task difficulty based on model performance. Reward functions evaluate model outputs across multiple dimensions (correctness, format, quality).

## Key Architecture

### Curriculum Learning (`infinite_rl/curriculum.py`)

**Core responsibility**: Manage task difficulty progression and orchestrate reward evaluation.

**Critical concepts**:
- **Sliding Window Success Rates**: Tracks last N evaluations (default: 50) per difficulty level using `deques`
- **GRPO Batching**: Groups multiple generations (default: 4) per prompt, processes at batch completion only
- **Leaky Gate Strategy**: During warmup (first 32 steps), partial credit (0.1) for proper format even if answer wrong
- **Per-Level Tracking**: Independent success windows per level (0-6), not per task type

**Key parameters that control level changes**:
- `success_rate_threshold` (default: 0.7): Advance when success rate exceeds this
- `demote_threshold` (default: 0.4): Demote only if success rate consistently below this AND variance < threshold
- `variance_threshold` (default: 0.15): Demotion requires stable (low variance) poor performance
- `level_change_cooldown` (default: 5): Minimum steps between level changes to prevent oscillation
- `warmup_step` (default: 32): Initial steps using only Level 0 tasks

**Important implementation details**:
- `_track_success_group()`: Uses PRIMARY scores (correctness) for curriculum decisions, separating format issues from ability
- `_update_level()`: Only checks CURRENT level's window; advancement based on success rate, demotion based on BOTH success rate AND variance
- `compute_reward()`: Accumulates scores in `grpo_batch_scores` dict until batch is complete, logs only once per batch
- `_apply_leaky_gate()`: Returns 1.0 (both tags + correct), 0.1 (both tags + wrong), or 0.0 (missing tags)

**Logging behavior**:
- Logs to JSONL (one entry per prompt/batch, not per generation)
- Includes `grpo_batch_size`, `grpo_primary_scores`, `grpo_combined_scores` arrays
- Single evaluation mode (`num_generations=1`) logs immediately without batch accumulation

### Reward Functions (`infinite_rl/reward_functions/`)

**Architecture**: All reward functions inherit from `RewardFunction` base class.

**Primary task correctness** (used for level advancement):
- `MathRewardFunction`: Symbolic equation evaluation using SymPy; returns 0.0 or 1.0
- `PuzzleRewardFunction`: Code execution against satisfaction functions; returns 0.0 or 1.0

**Auxiliary quality metrics** (optional, controlled by flags):
- `FormatRewardFunction`: Validates XML-like tags (`<think>`, `<answer>`); returns 1.0 (valid) or -1.0 (invalid)
- `ReasoningStepsRewardFunction`: Counts structural indicators ("first", "second", "finally"); returns 1.0 (3+), 0.7 (2), 0.5 (1), -1.0 (0)
- `LangConsistencyRewardFunction`: Checks reasoning language matches prompt; returns [−1, 1]
- `RepetitionRewardFunction`: Penalizes repeated phrases; returns [−0.02, 0]
- `LengthRewardFunction`: Length regularization; returns [0, 1]

**Combined score formula**:
```
combined = (1 - aux_weight) × primary + aux_weight × avg(auxiliary)
```
Default `aux_weight = 0.2` (80% primary, 20% auxiliary).

**Critical behavior**:
- Format is a hard gate during evaluation: checked BEFORE primary scoring
- Auxiliary rewards are included as-is regardless of primary task success/failure
- Negative penalties (repetition, lang_consistency) are always applied if detected
- Scores are clipped to [-1.0, 1.0] per function, then [-1.0, 1.0] after averaging

## When Making Changes

### Changes to Curriculum Progression Logic

1. **Modifying level advancement/demotion**: Update `_update_level()` and ensure changes respect the sliding window semantics
2. **Changing GRPO batch handling**: Verify batch accumulation in `compute_reward()` only logs at batch completion
3. **Warmup gate strategy**: Ensure both `_apply_leaky_gate()` and docstring stay synchronized
4. **Success tracking**: Primary scores drive curriculum; use `_track_success_group()` with primary_scores param

**Test coverage required**:
- `tests/test_curriculum.py`: Has 61+ tests covering warmup, level changes, GRPO batching, logging
- Run: `python -m pytest tests/test_curriculum.py -v` to validate

### Changes to Reward Functions

1. **New auxiliary reward**: Create class in `infinite_rl/reward_functions/`, inherit from `RewardFunction`
2. **Register reward**: Add to `get_reward_functions()` in `infinite_rl/reward_functions/__init__.py`
3. **Enable in curriculum**: Add `use_[reward_name]` parameter to `CurriculumLearning.__init__()` and initialize in `_initialize_aux_reward_functions()`
4. **Primary reward changes**: Affects curriculum advancement; requires test updates in `tests/test_reward_functions/test_*.py`

**Interface contract for all reward functions**:
- `compute_reward(task: Task, is_correct: bool = False) -> RewardFunctionScore`
- Return `RewardFunctionScore(score=..., info="...")` where score in [-1, 1]
- Handle timeouts gracefully (return 0.0 with error in info)

### Testing Checklist

After curriculum or reward function changes:

```bash
# Full test suite
python -m pytest tests/test_curriculum.py tests/test_reward_functions/ -v --tb=short

# Specific test for new/modified reward
python -m pytest tests/test_reward_functions/test_[reward_name].py -v

# Quick validation
python -m pytest tests/test_curriculum.py::TestCurriculumLearning -q
```

## Common Pitfalls

1. **GRPO batch logging**: Don't log per-generation; accumulate until batch complete. Log only once per prompt-batch.
2. **Sliding window semantics**: Each level tracks independently. Clearing window on level change is correct; prevents old data drift.
3. **Format vs correctness**: Format is hard gate during warmup; primary score is used for curriculum. Don't confuse the two.
4. **Auxiliary reward handling**: Auxiliary rewards are included as-is regardless of primary task success/failure. Always apply negative penalties.
5. **Variance for demotion**: Variance threshold only used for demotion (stable bad performance). Not for advancement (success rate alone).

## Documentation References

- [CURRICULUM.md](../../CURRICULUM.md): Complete parameter guide, tuning examples, monitoring instructions
- [infinite_rl/curriculum.py](../../infinite_rl/curriculum.py): `CurriculumLearning.__init__()` docstring lists all parameters
- [infinite_rl/reward_functions/](../../infinite_rl/reward_functions/): Each reward function has its own docstring

## File Structure

```
infinite_rl/
├── curriculum.py              # Main curriculum learning class
├── reward_functions/
│   ├── __init__.py           # get_reward_functions() registry
│   ├── reward_function.py    # RewardFunction base class
│   ├── math.py               # MathRewardFunction
│   ├── puzzle.py             # PuzzleRewardFunction
│   ├── format.py             # FormatRewardFunction
│   ├── reasoning_steps.py    # ReasoningStepsRewardFunction
│   ├── lang_consistency.py   # LangConsistencyRewardFunction
│   ├── repetition.py         # RepetitionRewardFunction
│   └── length.py             # LengthRewardFunction
└── session.py                # Session tracking task history
tests/
├── test_curriculum.py        # 61+ tests for curriculum logic, GRPO, logging
└── test_reward_functions/
    ├── test_math_reward_function.py
    ├── test_puzzle_reward_function.py
    ├── test_format_reward_function.py
    └── ...
```

## Debugging Tips

**Check curriculum state**:
```python
from infinite_rl.curriculum import CurriculumLearning
curriculum = CurriculumLearning(log_file="test.jsonl")
stats = curriculum.get_learning_stats()
print(f"Level: {stats['current_level']}, Success stats: {stats['sliding_window_stats']}")
```

**Inspect log entries**:
```bash
# Last 5 log entries (JSONL format)
tail -5 curriculum_log.jsonl | python -m json.tool
```

**GRPO batch debugging**:
- Check `grpo_batch_size` in log: should equal `num_generations` (default 4)
- All entries in `grpo_primary_scores` array should be 0.0 or 1.0 (binary correctness)
- `grpo_combined_scores` includes auxiliary penalties, typically [0.95, 1.0] for correct tasks

**Test a single reward function**:
```python
from infinite_rl.reward_functions import MathRewardFunction
from infinite_rl.task import Task

rf = MathRewardFunction()
task = Task(..., model_output="<think>...</think><answer>56</answer>")
score = rf.compute_reward(task)
print(f"Score: {score.score}, Info: {score.info}")
```
