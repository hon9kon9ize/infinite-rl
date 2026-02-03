# GRPO Refactoring: Clean Implementation Guide

## Overview

This guide implements **Option A (Clean Hierarchy)** with zero unused logic. We create a proper `Task → Generation` hierarchy and migrate away from old APIs immediately.

## Phase 1: Add Generation Class (30 mins)

### 1. Create `infinite_rl/generation.py`

```python
"""Generation data model for GRPO batches."""

import datetime
from dataclasses import dataclass, field
from typing import List

from .reward_functions import RewardFunctionScore


@dataclass
class Generation:
    """A single generation in a GRPO batch.
    
    Represents one model output with its rewards and score.
    """
    output: str
    rewards: List[RewardFunctionScore]
    primary_score: float
    created_at: datetime.datetime = field(default_factory=datetime.datetime.now)
    
    @property
    def is_correct(self) -> bool:
        """Whether this generation is considered correct."""
        return self.primary_score >= 0.5
    
    def to_dict(self) -> dict:
        """Convert to dict for logging."""
        return {
            "output": self.output,
            "rewards": [r.to_dict() for r in self.rewards],
            "primary_score": self.primary_score,
            "is_correct": self.is_correct,
            "created_at": self.created_at.isoformat(),
        }
```

### 2. Update `infinite_rl/task.py`

```python
# Add import
from .generation import Generation

class Task:
    def __init__(self, task_id: str, prompt: str, ...):
        # Existing fields
        self.task_id = task_id
        self.prompt = prompt
        # ... other existing fields ...
        
        # NEW: Clean generation hierarchy
        self.generations: List[Generation] = []
    
    def add_generation(self, output: str, rewards: List[RewardFunctionScore], 
                      primary_score: float) -> Generation:
        """Add a generation to this task."""
        gen = Generation(
            output=output,
            rewards=rewards,
            primary_score=primary_score
        )
        self.generations.append(gen)
        return gen
    
    @property
    def latest_generation(self) -> Optional[Generation]:
        """Get the most recent generation."""
        return self.generations[-1] if self.generations else None
    
    # TEMPORARY COMPATIBILITY PROPERTIES (remove after migration)
    @property
    def model_output(self) -> Optional[str]:
        """DEPRECATED: Use task.generations[-1].output instead."""
        latest = self.latest_generation
        return latest.output if latest else None
    
    @property
    def task_rewards(self) -> List[RewardFunctionScore]:
        """DEPRECATED: Use task.generations[-1].rewards instead."""
        latest = self.latest_generation
        return latest.rewards if latest else []
    
    @property
    def is_correct(self) -> Optional[bool]:
        """DEPRECATED: Use task.generations[-1].is_correct instead."""
        latest = self.latest_generation
        return latest.is_correct if latest else None
    
    def to_dict(self) -> dict:
        """Convert to dict for logging."""
        result = {
            # Existing fields...
            "task_id": self.task_id,
            "prompt": self.prompt,
            # ... other existing fields ...
            
            # NEW: All generations
            "generations": [g.to_dict() for g in self.generations],
        }
        return result
```

## Phase 2: Update CurriculumLearning (1 hour)

### 1. Remove GRPO batch dicts

```python
class CurriculumLearning:
    def __init__(self, ...):
        # ... existing init ...
        
        # REMOVE these dicts:
        # self.grpo_batch_primary_scores: Dict[str, List[float]] = {}
        # self.grpo_batch_outputs: Dict[str, List[str]] = {}
```

### 2. Update `compute_reward` method

```python
def compute_reward(self, task: Task, model_output: str, 
                  reward_functions: List[RewardFunction]) -> float:
    """Compute reward for a single generation."""
    
    # Get rewards for this generation
    rewards = []
    for rf in reward_functions:
        score = rf(task, model_output)
        rewards.append(score)
    
    # Calculate primary score (correctness)
    primary_score = self._calculate_primary_score(rewards)
    
    # ADD generation to task (instead of dicts)
    task.add_generation(model_output, rewards, primary_score)
    
    # Update curriculum progress
    self._update_curriculum_progress(task, primary_score)
    
    return primary_score
```

### 3. Update batch completion logic

```python
def _on_batch_complete(self, base_task: Task):
    """Called when a GRPO batch is complete."""
    
    # REMOVE dict cleanup:
    # if base_task.task_id in self.grpo_batch_primary_scores:
    #     del self.grpo_batch_primary_scores[base_task.task_id]
    # if base_task.task_id in self.grpo_batch_outputs:
    #     del self.grpo_batch_outputs[base_task.task_id]
    
    # Task now owns all its generations - no cleanup needed!
    
    # Log all generations
    self._log_batch_generations(base_task)
```

### 4. Add batch logging method

```python
def _log_batch_generations(self, task: Task):
    """Log all generations in a batch."""
    if not self.logger:
        return
    
    for i, gen in enumerate(task.generations):
        self.logger.log({
            "task_id": task.task_id,
            "generation_idx": i,
            "model_output": gen.output,
            "primary_score": gen.primary_score,
            "rewards": [r.to_dict() for r in gen.rewards],
            "is_correct": gen.is_correct,
            "batch_size": len(task.generations),
        })
```

## Phase 3: Update Session Class (30 mins)

### 1. Add batch query methods

```python
class Session:
    def __init__(self):
        self.tasks: List[Task] = []
    
    def get_batch_data(self, task_id: str) -> Optional[List[dict]]:
        """Get all generations for a task."""
        task = self.get_task(task_id)
        if not task:
            return None
        return [g.to_dict() for g in task.generations]
    
    def get_batch_stats(self, task_id: str) -> Optional[dict]:
        """Get statistics for all generations in a task."""
        task = self.get_task(task_id)
        if not task or not task.generations:
            return None
        
        scores = [g.primary_score for g in task.generations]
        return {
            "count": len(scores),
            "avg_score": sum(scores) / len(scores),
            "min_score": min(scores),
            "max_score": max(scores),
            "correct_count": sum(1 for g in task.generations if g.is_correct),
        }
```

## Phase 4: Update Training Script (30 mins)

### 1. Update wandb logging

```python
# In train.py, update logging to use new structure
def log_curriculum_metrics(curriculum: CurriculumLearning, step: int):
    """Log curriculum metrics to wandb."""
    
    # Existing metrics...
    wandb.log({
        "curriculum/level": curriculum.current_level,
        "curriculum/success_rate": curriculum.get_success_rate(),
        # ... other existing metrics ...
    })
    
    # NEW: Log generation statistics across recent tasks
    recent_tasks = curriculum.session.tasks[-10:]  # Last 10 tasks
    if recent_tasks:
        all_scores = []
        for task in recent_tasks:
            all_scores.extend([g.primary_score for g in task.generations])
        
        if all_scores:
            wandb.log({
                "curriculum/generation_avg_score": sum(all_scores) / len(all_scores),
                "curriculum/generation_count": len(all_scores),
                "curriculum/tasks_with_multiple_generations": sum(
                    1 for t in recent_tasks if len(t.generations) > 1
                ),
            })
```

## Phase 5: Migration Script (15 mins)

### 1. Find all old API usages

```bash
# Find files that need updating
grep -r "task\.model_output\|task\.task_rewards\|task\.is_correct" --include="*.py" .
```

### 2. Create migration script

```python
#!/usr/bin/env python3
"""Migration script to update old Task APIs to new Generation hierarchy."""

import os
import re
import subprocess

def migrate_file(filepath: str):
    """Migrate a single file."""
    
    with open(filepath, 'r') as f:
        content = f.read()
    
    # Replace old APIs with new ones
    replacements = [
        (r'task\.model_output', r'task.generations[-1].output'),
        (r'task\.task_rewards', r'task.generations[-1].rewards'),
        (r'task\.is_correct', r'task.generations[-1].is_correct'),
    ]
    
    for old, new in replacements:
        content = re.sub(old, new, content)
    
    with open(filepath, 'w') as f:
        f.write(content)
    
    print(f"Migrated {filepath}")

def main():
    """Run migration on all Python files."""
    
    # Find all Python files
    result = subprocess.run(
        ['find', '.', '-name', '*.py', '-not', '-path', './venv/*'],
        capture_output=True, text=True
    )
    
    files = result.stdout.strip().split('\n')
    
    for filepath in files:
        if os.path.exists(filepath):
            migrate_file(filepath)
    
    print("Migration complete!")

if __name__ == "__main__":
    main()
```

## Phase 6: Remove Deprecated Properties (15 mins)

### 1. Remove compatibility properties from Task

```python
class Task:
    # REMOVE these after migration:
    # @property
    # def model_output(self) -> Optional[str]: ...
    # @property  
    # def task_rewards(self) -> List[RewardFunctionScore]: ...
    # @property
    # def is_correct(self) -> Optional[bool]: ...
```

## Phase 7: Update Tests (1 hour)

### 1. Update existing tests

```python
def test_task_generation_tracking():
    """Test that Task properly tracks generations."""
    task = Task("test_task", "test prompt")
    
    # Add generations
    task.add_generation("output1", [], 0.8)
    task.add_generation("output2", [], 0.6)
    
    assert len(task.generations) == 2
    assert task.generations[0].output == "output1"
    assert task.generations[1].output == "output2"
    assert task.generations[0].primary_score == 0.8
    
    # Test latest generation
    assert task.latest_generation.output == "output2"

def test_curriculum_no_dicts():
    """Test that CurriculumLearning no longer uses dicts."""
    curriculum = CurriculumLearning(...)
    
    # Should not have these attributes
    assert not hasattr(curriculum, 'grpo_batch_primary_scores')
    assert not hasattr(curriculum, 'grpo_batch_outputs')

def test_session_batch_queries():
    """Test Session batch query methods."""
    session = Session()
    
    task = Task("test", "prompt")
    task.add_generation("out1", [], 0.7)
    task.add_generation("out2", [], 0.9)
    session.add_task(task)
    
    # Test batch data
    batch_data = session.get_batch_data("test")
    assert len(batch_data) == 2
    assert batch_data[0]["output"] == "out1"
    
    # Test batch stats
    stats = session.get_batch_stats("test")
    assert stats["count"] == 2
    assert stats["avg_score"] == 0.8
    assert stats["correct_count"] == 2
```

## Testing Strategy

### 1. Unit Tests
- Test `Generation` class
- Test `Task.add_generation()`
- Test `Session.get_batch_data()`
- Test curriculum without dicts

### 2. Integration Tests
- Test full GRPO batch workflow
- Test logging captures all generations
- Test wandb logging includes generation stats

### 3. Regression Tests
- Ensure old functionality still works during migration
- Test that compatibility properties work
- Test migration script doesn't break syntax

## Rollback Plan

If something goes wrong:

1. **Phase 1-3**: Revert commits, restore old dict-based approach
2. **Phase 4-5**: Use git to undo migration script changes
3. **Phase 6-7**: Restore compatibility properties temporarily

## Success Criteria

✅ **Zero unused logic**: No old APIs or redundant fields  
✅ **Clean hierarchy**: `Task → generations → rewards/scores`  
✅ **All generations logged**: JSONL contains full batch data  
✅ **Queryable history**: `session.get_batch_data(task_id)` works  
✅ **Tests pass**: All existing and new tests pass  
✅ **Performance same**: No regression in training speed  

## Timeline Summary

- **Phase 1**: 30 mins (add Generation class)
- **Phase 2**: 1 hour (update CurriculumLearning)  
- **Phase 3**: 30 mins (update Session)
- **Phase 4**: 30 mins (update training script)
- **Phase 5**: 15 mins (run migration script)
- **Phase 6**: 15 mins (remove deprecated properties)
- **Phase 7**: 1 hour (update tests)

**Total: ~4 hours** with zero unused logic.

### New Task Structure
```python
class Task:
    def __init__(self, ...):
        self.task_id = task_id
        self.prompt = prompt
        self.generations: List[Generation] = []  # Clean hierarchy
        
class Generation:
    def __init__(self):
        self.output: str
        self.rewards: List[RewardFunctionScore]
        self.primary_score: float
        self.created_at: datetime
```

### Migration Path
1. **Phase 1**: Add `Generation` class and `task.generations` list
2. **Phase 2**: Update all code to use `task.generations[-1].output` instead of `task.model_output`
3. **Phase 3**: Remove old fields
4. **Phase 4**: Simplify to direct access: `task.generations[0].output`

## Recommendation: Option A with Migration Plan

**Why Option A?**
- ✅ Zero redundancy (no old APIs)
- ✅ Clean hierarchy (Task → Generations)
- ✅ Natural data model
- ✅ Future extensible

**Migration Plan:**
1. **Week 1**: Add `Generation` class and `task.generations` list
2. **Week 2**: Update all `task.model_output` → `task.generations[-1].output`
3. **Week 3**: Update all `task.task_rewards` → `task.generations[-1].rewards`
4. **Week 4**: Remove old fields, simplify to direct access

**Breaking Changes Required:**
- `task.model_output` → `task.generations[-1].output`
- `task.task_rewards` → `task.generations[-1].rewards`  
- `task.is_correct` → `task.generations[-1].primary_score >= 0.5`

But the result is **zero unused logic**.

## Implementation: Option A

### Step 1: Add Generation Class
```python
@dataclass
class Generation:
    """A single generation in a GRPO batch."""
    output: str
    rewards: List[RewardFunctionScore]
    primary_score: float
    created_at: datetime.datetime = field(default_factory=datetime.datetime.now)
    
    @property
    def is_correct(self) -> bool:
        return self.primary_score >= 0.5
```

### Step 2: Update Task Class
```python
class Task:
    def __init__(self, ...):
        # Existing fields
        self.task_id = task_id
        self.prompt = prompt
        # ... other fields ...
        
        # NEW: Clean hierarchy
        self.generations: List[Generation] = []
    
    def add_generation(self, output: str, rewards: List[RewardFunctionScore], 
                      primary_score: float) -> Generation:
        """Add a generation to this task."""
        gen = Generation(
            output=output,
            rewards=rewards,
            primary_score=primary_score
        )
        self.generations.append(gen)
        return gen
    
    @property
    def latest_generation(self) -> Optional[Generation]:
        """Get the most recent generation."""
        return self.generations[-1] if self.generations else None
    
    @property
    def model_output(self) -> Optional[str]:  # DEPRECATED during migration
        """Temporary compatibility - remove after migration."""
        latest = self.latest_generation
        return latest.output if latest else None
    
    @property
    def task_rewards(self) -> List[RewardFunctionScore]:  # DEPRECATED
        """Temporary compatibility - remove after migration."""
        latest = self.latest_generation
        return latest.rewards if latest else []
    
    @property
    def is_correct(self) -> Optional[bool]:  # DEPRECATED
        """Temporary compatibility - remove after migration."""
        latest = self.latest_generation
        return latest.is_correct if latest else None
```

### Step 3: Update CurriculumLearning
```python
# Remove dicts entirely
# Replace with:
task.add_generation(output, rewards, score)
```

### Step 4: Migration Script
```python
# Run this to update all code:
# find . -name "*.py" -exec sed -i 's/task\.model_output/task.generations[-1].output/g' {} \;
# find . -name "*.py" -exec sed -i 's/task\.task_rewards/task.generations[-1].rewards/g' {} \;
# find . -name "*.py" -exec sed -i 's/task\.is_correct/task.generations[-1].is_correct/g' {} \;
```

### Step 5: Remove Deprecated Properties
After all code updated, remove the `@property` methods.

## Result: Zero Unused Logic

**Before (Redundant):**
```
Task
├── model_output: str  ← OLD
├── task_rewards: List ← OLD  
├── generations: List[Generation] ← NEW
└── generation_outputs: List[str] ← REDUNDANT
```

**After (Clean):**
```
Task
├── prompt: str
├── generations: List[Generation]
│   ├── [0]: Generation
│   │   ├── output: str
│   │   ├── rewards: List
│   │   └── primary_score: float
│   └── [1]: Generation
│       └── ...
└── Methods:
    ├── add_generation()
    ├── latest_generation @property
    └── get_all_generations()
```

## Timeline: 2 Weeks

**Week 1: Add new structure**
- Add `Generation` class
- Add `task.generations` list
- Add compatibility properties
- Update curriculum to use new structure

**Week 2: Migrate and clean**
- Update all code to use new APIs
- Remove compatibility properties
- Remove old fields
- Update tests

**Result:** Clean, maintainable code with zero unused logic.

## Questions?

**Q: How do I find all usages of old APIs?**
A: `grep -r "task\.model_output\|task\.task_rewards\|task\.is_correct" .`

**Q: What if I miss some code?**
A: Compatibility properties will catch it during testing.

**Q: Can I do this incrementally?**
A: Yes - add new structure first, then migrate module by module.

**Q: Performance impact?**
A: None - same data, cleaner access patterns.</content>
<parameter name="filePath">/Users/josephcheng/Projects/rl-data-geneator/docs/GRPO_REFACTORING_CLEAN_APPROACH.md