# GRPO Refactoring: Clean Approach (No Unused Logic)

## Problem with Backward Compatibility

You're right - Option C introduces redundancy:
- `task.model_output` (old, single string)
- `task.generation_outputs` (new, list of strings)
- `task.task_rewards` (old, mixed from all generations)
- `task.generation_rewards` (new, per-generation rewards)

This creates unused logic that needs maintenance.

## Solution: Option C+ (Clean Migration)

Use Option C's structure but **immediately deprecate old APIs** after migration.

### Phase 1: Add New APIs (Clean)
```python
class Task:
    def __init__(self, ...):
        # NEW: Only generation tracking (no old fields)
        self.generation_outputs: List[str] = []
        self.generation_rewards: List[List[RewardFunctionScore]] = []
        self.primary_scores: List[float] = []
        
        # DEPRECATED: Remove these immediately after migration
        # self.model_output: Optional[str] = None  # ← REMOVE
        # self.task_rewards: List[RewardFunctionScore] = []  # ← REMOVE
```

### Phase 2: Update All Usage (Breaking but Clean)
```python
# BEFORE (scattered):
self.grpo_batch_primary_scores[base_task_id].append(score)
self.grpo_batch_outputs[base_task_id].append(task.model_output)

# AFTER (clean):
task.add_generation(output, rewards, score)
```

### Phase 3: Remove Old APIs (Clean)
After all code updated, remove:
- `task.model_output`
- `task.task_rewards` 
- `task.is_correct` (derived from latest primary_score)

## Alternative: Option A (Clean Hierarchy)

If you want zero redundancy, use Option A with a clear migration:

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
- `task.is_correct` → `task.generations[-1].is_correct`

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
A: None - same data, cleaner access patterns.