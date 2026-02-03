# GRPO Batch Refactoring Guide

## Problem Statement

Currently, GRPO batch state is scattered across multiple places:
- `CurriculumLearning.grpo_batch_primary_scores[task_id]` → list of scores
- `CurriculumLearning.grpo_batch_outputs[task_id]` → list of outputs
- `Task.model_output` → single (latest) output only
- `Task.task_rewards` → mixed rewards from all generations

**This makes it impossible to:**
- Query: "What was the score for generation #2?"
- Reconstruct: Which reward corresponds to which generation?
- Analyze: How did the model improve across generations?
- Debug: Which generation had what state?

## Architecture Options

### Option A: Task → Generation Hierarchy (Clean but Breaking)

```
Task (prompt)
├── Generation 0
│   ├── model_output: "..."
│   ├── rewards: [RewardFunctionScore, ...]
│   └── primary_score: 0.8
├── Generation 1
│   ├── model_output: "..."
│   ├── rewards: [RewardFunctionScore, ...]
│   └── primary_score: 1.0
└── Generation 2
    └── ...
```

**Pros:**
- ✅ Natural hierarchy
- ✅ Clean API: `task.generations[0].score`
- ✅ No scattered state
- ✅ Easy to serialize

**Cons:**
- ❌ Breaking change (big refactor)
- ❌ `task.model_output` becomes invalid concept

---

### Option B: Session-Managed Batches (Separation of Concerns)

```
Session
├── tasks: {task_id: Task}
└── grpo_batches: {task_id: GRPOBatch}
    └── Task 123
        ├── generations: ["output1", "output2", ...]
        ├── primary_scores: [0.8, 1.0, ...]
        └── is_complete: bool
```

**Pros:**
- ✅ Session owns all state
- ✅ Minimal Task changes
- ✅ Clean separation

**Cons:**
- ❌ Two places to look (Task + GRPOBatch)
- ❌ Need two lookups: `get_task()` + `get_batch()`

---

### Option C: Task-Based Tracking (Recommended - Hybrid)

```
Task (enhanced)
├── prompt: str
├── model_output: Optional[str]  # Latest (backward compat)
├── task_rewards: List[RewardFunctionScore]  # Latest (backward compat)
│
├── generation_outputs: List[str]  # NEW: All outputs
├── generation_rewards: List[List[RewardFunctionScore]]  # NEW: Per-generation rewards
├── primary_scores: List[float]  # NEW: Per-generation primary scores
│
└── Methods:
    ├── add_generation(output, rewards, score) → int
    ├── get_generation(idx) → Dict
    ├── get_all_generations() → List[Dict]
    └── (backward compat: model_output, task_rewards auto-updated)
```

**Pros:**
- ✅ Backward compatible
- ✅ Task owns everything
- ✅ Clear mapping: generation_i ↔ rewards_i ↔ score_i
- ✅ Session can access batch data
- ✅ Migration path (deprecate old APIs later)

**Cons:**
- ⚠️ Slight redundancy (two ways to access latest)
- ⚠️ Coexistence of old/new APIs during transition

---

## Recommended Approach: Option C

### Why Option C?

1. **Backward Compatible**: Existing code still works
2. **Clear Ownership**: Task holds all its generations
3. **Session Integration**: Enables `session.get_batch_data(task_id)`
4. **Single Source of Truth**: No scattered dicts
5. **Low Risk**: Can roll out incrementally
6. **Future Proof**: Can deprecate old API later

### Implementation: 3 Phases

#### Phase 1: Extend Task Class

```python
class Task:
    def __init__(self, ...):
        # Existing fields (backward compat)
        self.model_output: Optional[str] = None
        self.task_rewards: List[RewardFunctionScore] = []
        self.is_correct: Optional[bool] = None
        
        # NEW: Generation tracking
        self.generation_outputs: List[str] = []
        self.generation_rewards: List[List[RewardFunctionScore]] = []
        self.primary_scores: List[float] = []
    
    def add_generation(
        self, 
        output: str, 
        rewards: List[RewardFunctionScore], 
        primary_score: float
    ) -> int:
        """Add a generation and return its index."""
        idx = len(self.generation_outputs)
        self.generation_outputs.append(output)
        self.generation_rewards.append(rewards)
        self.primary_scores.append(primary_score)
        
        # Backward compat: update "latest" fields
        self.model_output = output
        self.task_rewards = rewards
        # Update is_correct based on primary score
        self.is_correct = primary_score >= 0.5
        
        return idx
    
    def get_generation(self, idx: int) -> Dict[str, Any]:
        """Get a single generation's data."""
        if idx < 0 or idx >= len(self.generation_outputs):
            raise IndexError(f"Generation {idx} out of range")
        
        return {
            "generation_idx": idx,
            "output": self.generation_outputs[idx],
            "rewards": self.generation_rewards[idx],
            "primary_score": self.primary_scores[idx],
            "is_correct": self.primary_scores[idx] >= 0.5,
        }
    
    def get_all_generations(self) -> List[Dict[str, Any]]:
        """Get all generations for batch analysis."""
        return [self.get_generation(i) for i in range(len(self.generation_outputs))]
    
    def get_batch_stats(self) -> Dict[str, Any]:
        """Get GRPO batch statistics."""
        if not self.primary_scores:
            return {"batch_size": 0}
        
        return {
            "batch_size": len(self.primary_scores),
            "primary_scores": self.primary_scores,
            "avg_score": sum(self.primary_scores) / len(self.primary_scores),
            "max_score": max(self.primary_scores),
            "min_score": min(self.primary_scores),
            "num_correct": sum(1 for s in self.primary_scores if s >= 0.5),
        }
```

**Migration checklist:**
- [ ] Add the three new list fields
- [ ] Implement `add_generation()`, `get_generation()`, `get_all_generations()`
- [ ] Update `to_dict()` to include generation data
- [ ] Tests pass (backward compat)

---

#### Phase 2: Refactor CurriculumLearning

**Remove:**
```python
# DELETE these instance variables
self.grpo_batch_primary_scores: Dict[str, List[float]] = {}
self.grpo_batch_outputs: Dict[str, List[str]] = {}
```

**Update compute_reward():**
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

**Update batch completion logic:**
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

**Update batch completion logic (continued):**
```python
# At the end, no cleanup needed:
# DELETE: del self.grpo_batch_primary_scores[base_task_id]
# DELETE: del self.grpo_batch_outputs[base_task_id]
# Data stays in task, which is added to session anyway
```

**Migration checklist:**
- [ ] Remove the two dict instance variables
- [ ] Update all `append()` calls to `task.add_generation()`
- [ ] Update all logging to use `task.get_all_generations()`
- [ ] Update `_track_success_group()` to accept `task` instead of `primary_scores`
- [ ] Tests pass

---

#### Phase 3: Enhance Session

```python
class Session:
    # Existing...
    
    def get_batch_data(self, task_id: str) -> Optional[List[Dict]]:
        """Get all generations and scores for a GRPO batch."""
        task = self.get_task(task_id)
        if task is None:
            return None
        return task.get_all_generations()
    
    def get_batch_stats(self, task_id: str) -> Optional[Dict]:
        """Get statistics about a batch (avg score, size, etc.)"""
        task = self.get_task(task_id)
        if task is None:
            return None
        return task.get_batch_stats()
```

**Migration checklist:**
- [ ] Add `get_batch_data()` method
- [ ] Add `get_batch_stats()` method
- [ ] Add tests for these new methods

---

## Code Change Examples

### Example 1: Adding a Generation

```python
# In CurriculumLearning.compute_reward() or wherever generations are created

# Get the task
task = self.session.get_task(task_id)

# ... compute rewards ...

# BEFORE:
self.grpo_batch_primary_scores[base_task_id].append(score)
self.grpo_batch_outputs[base_task_id].append(task.model_output)

# AFTER:
generation_idx = task.add_generation(
    output=task.model_output,
    rewards=task_rewards,
    primary_score=score
)
print(f"Added generation {generation_idx} to {task_id}")
```

### Example 2: Processing a Complete Batch

```python
# BEFORE:
if len(self.grpo_batch_primary_scores[base_task_id]) >= self.num_generations:
    primary_scores = self.grpo_batch_primary_scores[base_task_id]
    self._track_success_group(task.level, primary_scores)
    
    log_entry["grpo_batch_size"] = len(primary_scores)
    log_entry["grpo_primary_scores"] = primary_scores
    log_entry["grpo_model_outputs"] = self.grpo_batch_outputs[base_task_id]
    
    del self.grpo_batch_primary_scores[base_task_id]
    del self.grpo_batch_outputs[base_task_id]

# AFTER:
if len(task.generation_outputs) >= self.num_generations:
    batch_data = task.get_all_generations()
    primary_scores = [g["primary_score"] for g in batch_data]
    self._track_success_group(task.level, primary_scores)
    
    log_entry["grpo_batch_size"] = len(batch_data)
    log_entry["grpo_primary_scores"] = primary_scores
    log_entry["grpo_model_outputs"] = [g["output"] for g in batch_data]
    log_entry["grpo_generation_data"] = batch_data  # Full generation info!
    
    # No cleanup needed - task owns the data
```

### Example 3: Querying Generation Data

```python
# NEW capability with Session integration

# Get a task and its complete batch
task = session.get_task("task_123")

# Access generations directly
for gen in task.get_all_generations():
    print(f"Generation {gen['generation_idx']}")
    print(f"  Score: {gen['primary_score']}")
    print(f"  Output: {gen['output'][:50]}...")
    print(f"  Rewards: {[r.reward_function_name for r in gen['rewards']]}")

# Or through session
batch = session.get_batch_data("task_123")
stats = session.get_batch_stats("task_123")
print(f"Batch size: {stats['batch_size']}, Avg score: {stats['avg_score']}")
```

---

## Benefits Per Component

### Task Class
- ✅ Owns complete generation history
- ✅ Clear generation-to-reward mapping
- ✅ Supports both single and batch queries
- ✅ Backward compatible

### CurriculumLearning
- ✅ No scattered state (dicts removed)
- ✅ Cleaner code (fewer instance vars)
- ✅ Less memory (no duplication)
- ✅ Easier to debug (all data in task)

### Session
- ✅ Can provide batch query APIs
- ✅ Enables analytics on full generation history
- ✅ Supports future features (replay, analysis)

### Logging & Analysis
- ✅ Complete generation-to-score trace
- ✅ Can compute per-generation statistics
- ✅ Supports GRPO-specific metrics
- ✅ Enables detailed reward function analysis

---

## Migration Path

### Pre-Refactor
```
CurriculumLearning
├── grpo_batch_primary_scores: Dict[task_id → scores]
├── grpo_batch_outputs: Dict[task_id → outputs]
└── compute_reward(...) → appends to dicts

Session
└── tasks: Dict[task_id → Task]
    └── Task
        ├── model_output: latest only
        └── task_rewards: mixed from all gens
```

### Post-Refactor
```
CurriculumLearning
├── ~~grpo_batch_primary_scores~~ (removed)
├── ~~grpo_batch_outputs~~ (removed)
└── compute_reward(...) → calls task.add_generation()

Session
└── tasks: Dict[task_id → Task]
    └── Task (enhanced)
        ├── model_output: latest (backward compat)
        ├── task_rewards: latest (backward compat)
        ├── generation_outputs: all ✨ NEW
        ├── generation_rewards: all ✨ NEW
        ├── primary_scores: all ✨ NEW
        └── Methods:
            ├── add_generation()
            ├── get_generation(idx)
            └── get_all_generations()
```

---

## Testing Strategy

### Unit Tests to Add

```python
def test_task_add_generation():
    task = Task(...)
    idx = task.add_generation(
        output="result",
        rewards=[RewardFunctionScore(0.8, "test", "")],
        primary_score=0.8
    )
    assert idx == 0
    assert task.model_output == "result"  # backward compat
    assert task.primary_scores[0] == 0.8

def test_task_get_all_generations():
    task = Task(...)
    task.add_generation("out1", [...], 0.8)
    task.add_generation("out2", [...], 1.0)
    
    gens = task.get_all_generations()
    assert len(gens) == 2
    assert gens[0]["primary_score"] == 0.8
    assert gens[1]["primary_score"] == 1.0

def test_session_get_batch_data():
    session = Session()
    task = Task(...)
    task.add_generation("out1", [...], 0.8)
    session.add_task(task)
    
    batch = session.get_batch_data(task.task_id)
    assert len(batch) == 2
```

### Integration Tests

- GRPO batch completion without dicts
- Logging includes all generations
- `_track_success_group()` works with task
- Backward compat: `task.model_output` still valid

### Regression Tests

- Run all existing tests (should pass unchanged)
- Verify JSONL log format includes generation data
- Check curriculum progression unaffected

---

## Future Enhancements

After refactoring:

1. **Generation-level analytics**: "Which generations improved?"
2. **Per-function reward analysis**: "How did format/reasoning scores differ across generations?"
3. **GRPO policy insights**: "Which generations won the batch?"
4. **Curriculum replay**: "Show me all generations for successful prompts at this level"
5. **Reward function debugging**: "Which gen got lowest score from format function?"

---

## Risk Assessment

| Risk | Likelihood | Mitigation |
|------|------------|-----------|
| Breaking existing code | **Low** | Backward compat: `model_output`, `task_rewards` still work |
| Lost batch data | **Low** | Data moves to Task (safer location) |
| Performance regression | **Very Low** | Fewer dicts, same complexity |
| Test failures | **Medium** | Comprehensive test suite updates |

**Recommendation**: Roll out Phase 1 + 2 together, verify tests pass, then Phase 3.
