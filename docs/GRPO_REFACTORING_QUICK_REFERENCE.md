# GRPO Refactoring: Quick Reference

## TL;DR - Recommended Solution: Option C (Hybrid)

Move GRPO batch state **FROM** scattered CurriculumLearning dicts **TO** Task class with generation tracking.

### Key Changes

**Task Class:** Add generation tracking
```python
task.add_generation(output, rewards, primary_score)  # Returns generation_idx
task.get_all_generations()  # Returns list of all generations
task.get_batch_stats()      # Returns batch statistics
```

**CurriculumLearning:** Remove dicts
```python
# DELETE:
self.grpo_batch_primary_scores  # Dict: task_id → [scores]
self.grpo_batch_outputs        # Dict: task_id → [outputs]

# ADD calls to:
task.add_generation(output, rewards, score)
```

**Session:** Enable batch queries
```python
session.get_batch_data(task_id)    # Get all generations
session.get_batch_stats(task_id)   # Get batch stats
```

---

## Why This Solution?

| Aspect | Current | After Refactor |
|--------|---------|-----------------|
| **Batch state location** | 2 dicts in CurriculumLearning | Task owns it |
| **Generation-to-score mapping** | Lost/implicit | Explicit (index-based) |
| **Memory management** | Manual cleanup | Automatic (with task) |
| **Session access** | N/A | `get_batch_data(task_id)` |
| **Backward compat** | N/A | ✅ Yes |
| **Breaking changes** | N/A | ❌ None |

---

## Three Implementation Phases

### Phase 1: Extend Task (1-2 hours)
- Add 3 new lists to Task: `generation_outputs`, `generation_rewards`, `primary_scores`
- Add methods: `add_generation()`, `get_generation()`, `get_all_generations()`
- Update `to_dict()` for logging
- Tests pass (backward compat guaranteed)

### Phase 2: Refactor CurriculumLearning (2-3 hours)
- Remove `self.grpo_batch_primary_scores` dict
- Remove `self.grpo_batch_outputs` dict
- Replace all `append()` calls with `task.add_generation()`
- Update logging to use `task.get_all_generations()`
- Remove batch cleanup code
- Tests pass

### Phase 3: Enhance Session (30 minutes)
- Add `get_batch_data(task_id)` method
- Add `get_batch_stats(task_id)` method
- Tests pass

**Total: ~4 hours of development, minimal risk**

---

## Current Problems This Solves

1. **"What was the score for generation #2?"**
   - Current: Dig through CurriculumLearning dicts
   - After: `task.get_generation(2)["primary_score"]`

2. **"Which reward goes with which generation?"**
   - Current: Unknown (mixed in task_rewards)
   - After: `task.generation_rewards[i]` for generation `i`

3. **"Where is the GRPO batch state?"**
   - Current: Split across CurriculumLearning AND Task
   - After: All in Task (single source of truth)

4. **"Can I analyze all generations after training?"**
   - Current: No (dicts cleaned up)
   - After: Yes (`session.get_batch_data(task_id)`)

5. **"Does this break existing code?"**
   - Current: N/A
   - After: No (backward compatible)

---

## Implementation Checklist

### Phase 1: Task Class

- [ ] Add `generation_outputs: List[str] = []`
- [ ] Add `generation_rewards: List[List[RewardFunctionScore]] = []`
- [ ] Add `primary_scores: List[float] = []`
- [ ] Implement `add_generation(output, rewards, score) → int`
- [ ] Implement `get_generation(idx) → Dict`
- [ ] Implement `get_all_generations() → List[Dict]`
- [ ] Implement `get_batch_stats() → Dict`
- [ ] Update `to_dict()` to include generations
- [ ] Update docstrings
- [ ] Run tests (should all pass)

### Phase 2: CurriculumLearning

- [ ] Delete `self.grpo_batch_primary_scores: Dict[str, List[float]] = {}`
- [ ] Delete `self.grpo_batch_outputs: Dict[str, List[str]] = {}`
- [ ] In `compute_reward()`: Replace `append()` calls with `task.add_generation()`
- [ ] In batch completion: Replace dict access with `task.get_all_generations()`
- [ ] Remove batch cleanup code (`del self.grpo_batch_*[base_task_id]`)
- [ ] Update logging to use `task.get_all_generations()`
- [ ] Update `_track_success_group()` signature if needed
- [ ] Run tests (should all pass)
- [ ] Run curriculum tests specifically

### Phase 3: Session

- [ ] Add `get_batch_data(task_id) → Optional[List[Dict]]`
- [ ] Add `get_batch_stats(task_id) → Optional[Dict]`
- [ ] Add docstrings
- [ ] Run tests

---

## Code Snippets Ready to Use

### Task.add_generation() Implementation
```python
def add_generation(
    self, 
    output: str, 
    rewards: List[RewardFunctionScore], 
    primary_score: float
) -> int:
    """Add a generation to this task's batch.
    
    Args:
        output: Model output string
        rewards: All reward scores for this generation
        primary_score: Primary correctness score
        
    Returns:
        Generation index (0-based)
    """
    idx = len(self.generation_outputs)
    self.generation_outputs.append(output)
    self.generation_rewards.append(rewards)
    self.primary_scores.append(primary_score)
    
    # Backward compat: update "latest" fields
    self.model_output = output
    self.task_rewards = rewards
    self.is_correct = primary_score >= 0.5
    
    return idx
```

### CurriculumLearning.compute_reward() Update
```python
# BEFORE:
self.grpo_batch_primary_scores[base_task_id].append(score)
self.grpo_batch_outputs[base_task_id].append(task.model_output)

# AFTER:
task.add_generation(
    output=task.model_output,
    rewards=task_rewards,
    primary_score=score
)
```

### Batch Completion Update
```python
# BEFORE:
primary_scores = self.grpo_batch_primary_scores[base_task_id]
log_entry["grpo_batch_size"] = len(primary_scores)
log_entry["grpo_primary_scores"] = primary_scores
log_entry["grpo_model_outputs"] = self.grpo_batch_outputs[base_task_id]

# AFTER:
batch_data = task.get_all_generations()
log_entry["grpo_batch_size"] = len(batch_data)
log_entry["grpo_primary_scores"] = [g["primary_score"] for g in batch_data]
log_entry["grpo_model_outputs"] = [g["output"] for g in batch_data]
```

---

## Testing Examples

```python
def test_generation_tracking():
    """Test that generations are properly tracked."""
    task = Task(task_id="test_1", ...)
    
    # Add first generation
    idx0 = task.add_generation(
        output="solution1",
        rewards=[RewardFunctionScore(0.8, "primary", "correct")],
        primary_score=0.8
    )
    assert idx0 == 0
    
    # Add second generation
    idx1 = task.add_generation(
        output="solution2",
        rewards=[RewardFunctionScore(1.0, "primary", "correct")],
        primary_score=1.0
    )
    assert idx1 == 1
    
    # Verify batch data
    batch = task.get_all_generations()
    assert len(batch) == 2
    assert batch[0]["primary_score"] == 0.8
    assert batch[1]["primary_score"] == 1.0
    
    # Verify backward compat
    assert task.model_output == "solution2"  # Latest
    assert task.primary_scores == [0.8, 1.0]

def test_session_batch_query():
    """Test that session can query batch data."""
    session = Session()
    task = Task(task_id="test_1", ...)
    task.add_generation("out1", [...], 0.8)
    task.add_generation("out2", [...], 1.0)
    session.add_task(task)
    
    batch = session.get_batch_data("test_1")
    assert batch is not None
    assert len(batch) == 2
    
    stats = session.get_batch_stats("test_1")
    assert stats["batch_size"] == 2
    assert stats["avg_score"] == 0.9
```

---

## Files to Modify

1. **infinite_rl/task.py** (Phase 1)
   - Add fields: `generation_outputs`, `generation_rewards`, `primary_scores`
   - Add methods: `add_generation()`, `get_generation()`, `get_all_generations()`, `get_batch_stats()`
   - Update `to_dict()`

2. **infinite_rl/curriculum.py** (Phase 2)
   - Remove: `self.grpo_batch_primary_scores`, `self.grpo_batch_outputs`
   - Update: `compute_reward()` method
   - Update: Batch completion logic
   - Update: Logging code

3. **infinite_rl/session.py** (Phase 3)
   - Add: `get_batch_data(task_id)`
   - Add: `get_batch_stats(task_id)`

4. **tests/test_task.py** (All phases)
   - Add generation tracking tests

5. **tests/test_curriculum.py** (Phase 2)
   - Update GRPO batch tests
   - Verify no dict references remain

---

## Backward Compatibility Guarantee

```python
# These continue to work exactly as before:
task.model_output        # Gets latest output
task.task_rewards        # Gets latest rewards
task.is_correct          # Gets latest correctness
task.to_dict()          # Includes backward compat fields

# These are new:
task.generation_outputs  # All outputs (plural)
task.generation_rewards  # All rewards (plural)
task.primary_scores      # All scores (plural)
task.add_generation()    # New method
task.get_all_generations()  # New method
session.get_batch_data()  # New method
```

**Zero breaking changes to existing APIs.**

---

## Performance Impact

| Metric | Before | After | Change |
|--------|--------|-------|--------|
| Memory (GRPO) | 2 dicts + task | 3 lists in task | ~Same |
| Dict lookups | 2 per generation | 0 | ✅ Better |
| Cleanup overhead | Manual + risky | Automatic | ✅ Better |
| Generation queries | Impossible | O(1) | ✅ New |

**Conclusion:** No performance regression, slight improvement due to fewer dicts.

---

## Rollout Strategy

1. **Start with Phase 1**: Only Task changes
   - Low risk (backward compatible)
   - Can merge and test independently
   - No curriculum changes needed

2. **Merge Phase 1**: Let tests stabilize

3. **Do Phase 2**: CurriculumLearning refactor
   - Builds on Phase 1
   - Removes scattered state
   - Tests verify behavior unchanged

4. **Merge Phase 2**: Verify curriculum tests pass

5. **Do Phase 3**: Session enhancements
   - Purely additive
   - No breaking changes
   - Enables future features

**Total time**: ~4 hours, 3 merge points, minimal risk

---

## Questions to Consider

**Q: What if I need to access a generation that was never added?**
A: Use `task.get_generation(idx)` which raises `IndexError` with clear message.

**Q: Can I still use the old `task.model_output` API?**
A: Yes! It's backward compatible (returns latest generation).

**Q: Do I need to update all existing code?**
A: No! Only new code should use `task.add_generation()`. Phase 2 refactor does that.

**Q: What about training logs - will the format change?**
A: Enhanced format (more data) but backward compatible. Old parsers still work.

**Q: Can I roll back if something breaks?**
A: Yes! Each phase is independent. Rollback Phase 2 without touching Phase 1.

---

## Success Criteria

- ✅ All existing tests pass unchanged
- ✅ New generation tracking tests pass
- ✅ GRPO batch tests pass
- ✅ No dict dereference errors
- ✅ Logging includes all generation data
- ✅ Session can query batch data
- ✅ Curriculum progression unchanged
- ✅ Memory usage similar or better
