"""
REWARD ORDERING BUG FIX - SUMMARY

This document summarizes the bug fix for the reward ordering issue in GRPO training.
"""

# REWARD ORDERING BUG - ROOT CAUSE ANALYSIS

## The Bug

In `scripts/train.py`, the `reward_func()` was assigning rewards to wrong generations:

```python
# BEFORE (BROKEN)
grouped = defaultdict(list)  # Group completions by task_id
for i, completion in enumerate(completions):
    grouped[task_id].append(completion)  # ❌ Lose original index!

for task_id, completion_list in grouped.items():  # ❌ Dict iteration order!
    batch_scores = curriculum.compute_rewards(task_id, completion_texts)
    rewards_list.extend(batch_scores)  # ❌ Extend in wrong order
```

**Example of the bug in action:**

```
Input completions: [C0, C1, C2, C3]
Input metadata:     [A,  B,  A,  B]

Grouped:
  task_A: [C0, C2]  
  task_B: [C1, C3]

If dict iteration returns task_B first:
  Process task_B: get scores [S1, S3]
  Process task_A: get scores [S0, S2]
  
  rewards_list = [S1, S3, S0, S2]  ❌ WRONG!
  Expected:      [S0, S1, S2, S3]  ✓ Correct
```

## Impact

- Rewards got randomly shuffled based on task_id ordering
- GRPO algorithm received contradictory learning signals
- Model couldn't learn because good generations got bad scores and vice versa
- Training loss stayed near zero (no learning happening)
- Curriculum never advanced (insufficient learning signals)

## The Fix

### Solution Pattern

Track original indices explicitly, group by task, then reconstruct in original order:

```python
# AFTER (FIXED)
indexed_completions = []
grouped = defaultdict(list)

# Step 1: Track original index for each completion
for i, completion in enumerate(completions):
    indexed_completions.append((i, task_id, completion))
    grouped[task_id].append((i, completion))  # ✓ Keep index!

# Step 2: Process each task, tracking indices
reward_with_index = {}
for task_id, index_completion_list in grouped.items():
    indices, completions_for_task = zip(*index_completion_list)
    batch_scores = curriculum.compute_rewards(task_id, completions_for_task)
    for idx, score in zip(indices, batch_scores):
        reward_with_index[idx] = score  # ✓ Map back to original index

# Step 3: Reconstruct in original order
rewards_list = [reward_with_index.get(i, 0.0) for i in range(len(completions))]
```

### Why This Works

- `reward_with_index` dictionary is indexed by ORIGINAL position (0, 1, 2, ...)
- Final loop reconstructs by iterating 0 to N, guaranteeing correct order
- Works regardless of task_id grouping or dict iteration order

## Testing

### Unit Tests (11 tests, all passing)

File: `tests/test_reward_ordering.py`

- ✓ Single task, sequential order
- ✓ Multiple tasks, interleaved order
- ✓ Complex 3-way interleaving
- ✓ Without task metadata
- ✓ Duplicate task IDs
- ✓ Single completion
- ✓ Many generations (16+)
- ✓ Curriculum called correctly
- ✓ Output length matches input
- ✓ Worst case (reverse order)
- ✓ Error handling

### Integration Tests

File: `tests/test_integration_curriculum.py`

- ✓ Order preservation end-to-end
- ✓ Full curriculum flow with rewards
- ✓ Generation tracking

### Run Tests

```bash
# Unit tests
pytest tests/test_reward_ordering.py -v

# Integration tests
python tests/test_integration_curriculum.py

# Validation summary
python scripts/validate_reward_ordering.py
```

## Files Changed

- `scripts/train.py`: Fixed `create_curriculum_reward_func()` → `reward_func()`

## Expected Impact on Training

### Before Fix
- Loss stays ~0 (no learning signal)
- Curriculum never advances (insufficient signals)
- Model performance doesn't improve

### After Fix
- Loss should become meaningful (varied per batch)
- Curriculum starts advancing (clear learning signals)
- Model performance improves over time
- GRPO optimization works correctly

## Verification Steps

After applying this fix and retraining:

1. **Check training loss curve:**
   - Should see non-zero, varying loss
   - Not stuck at zero or constant

2. **Check curriculum progression:**
   ```python
   stats = curriculum.get_learning_stats()
   print(f"Current level: {stats['current_level']}")  # Should increase over time
   print(f"Success rate: {stats['sliding_window_stats']}")
   ```

3. **Check generation quality:**
   - Look at logged rewards
   - Should see mix of high and low scores (not all 0)
   - Good generations should have higher scores than bad ones

4. **Check W&B logs:**
   - `curriculum/current_level` should increase
   - `curriculum/mean_success_rate` should vary
   - Training metrics should show improvement

## Root Cause Why I Found This

1. **GRPO Invariant**: Trainer expects `rewards[i]` to match `completions[i]`
2. **Dictionary Iteration**: Python 3.7+ preserves insertion order, but when you group by multiple keys and iterate, they come out in insertion order of first key, not grouped order
3. **Missing Index Tracking**: Original code lost the index mapping during grouping
4. **Consequence**: Rewards got shuffled, breaking the invariant

## Lessons Learned

1. **Never lose index mapping** when grouping/reshuffling data
2. **Always reconstruct in original order** before returning to caller
3. **Test with interleaved data** to catch ordering bugs
4. **Trust the invariant** - if the trainer expects order, it matters!

---

**Status**: ✅ FIXED AND TESTED
**Risk Level**: Low (fix is localized and well-tested)
**Rollout Plan**: Safe to deploy immediately
