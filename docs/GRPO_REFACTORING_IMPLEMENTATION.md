# GRPO Refactoring: Implementation Code Reference

## Phase 1: Task Class Enhancement

### New Fields to Add

```python
class Task:
    def __init__(self, ...):
        # Existing fields (keep all)
        self.task_id = task_id
        self.task_name = task_name
        self.task_type = task_type
        self.level = level
        self.prompt = prompt
        self.expected_answer = expected_answer
        self.task_rewards: List[RewardFunctionScore] = task_rewards or []
        self.is_correct: Optional[bool] = None
        self.model_output: Optional[str] = model_output
        self.created_at: datetime.datetime = created_at or datetime.datetime.now()
        self.first_response_at: Optional[datetime.datetime] = first_response_at
        self.language: Optional[str] = language
        self.reasoning_language: Optional[str] = reasoning_language or language or "en"
        
        # NEW: Generation tracking (list-based)
        self.generation_outputs: List[str] = []  # model_output for each gen
        self.generation_rewards: List[List[RewardFunctionScore]] = []  # rewards for each gen
        self.primary_scores: List[float] = []  # primary score for each gen
```

### New Methods to Add

```python
def add_generation(
    self, 
    output: str, 
    rewards: List[RewardFunctionScore], 
    primary_score: float
) -> int:
    """Add a generation to this task's GRPO batch.
    
    This tracks the output, rewards, and primary score for a single generation.
    Automatically updates backward-compatible fields (model_output, task_rewards, is_correct)
    to reflect the latest generation.
    
    Args:
        output: Model output string for this generation
        rewards: All reward scores for this generation
        primary_score: Primary correctness score (typically 0.0 or 1.0)
    
    Returns:
        Generation index (0-based)
    
    Raises:
        TypeError: If output is not a string or rewards is not a list
    
    Examples:
        >>> task = Task(...)
        >>> idx = task.add_generation("<answer>42</answer>", [...], 1.0)
        >>> idx
        0
        >>> task.primary_scores
        [1.0]
        >>> task.model_output  # backward compat
        '<answer>42</answer>'
    """
    idx = len(self.generation_outputs)
    
    # Validate inputs
    if not isinstance(output, str):
        raise TypeError(f"output must be str, got {type(output)}")
    if not isinstance(rewards, list):
        raise TypeError(f"rewards must be list, got {type(rewards)}")
    
    # Add to generation lists
    self.generation_outputs.append(output)
    self.generation_rewards.append(rewards)
    self.primary_scores.append(primary_score)
    
    # Backward compatibility: update "latest" fields
    self.model_output = output
    self.task_rewards = rewards
    self.is_correct = primary_score >= 0.5
    
    return idx


def get_generation(self, idx: int) -> Dict[str, Any]:
    """Get a single generation's complete data.
    
    Args:
        idx: Generation index (0-based)
    
    Returns:
        Dictionary with generation_idx, output, rewards, primary_score, is_correct
    
    Raises:
        IndexError: If idx out of range
    
    Examples:
        >>> gen = task.get_generation(0)
        >>> gen["primary_score"]
        1.0
        >>> gen["output"]
        '<answer>42</answer>'
        >>> len(gen["rewards"])
        4  # number of reward functions
    """
    if idx < 0 or idx >= len(self.generation_outputs):
        raise IndexError(
            f"Generation index {idx} out of range [0, {len(self.generation_outputs)-1}]"
        )
    
    return {
        "generation_idx": idx,
        "output": self.generation_outputs[idx],
        "rewards": self.generation_rewards[idx],
        "primary_score": self.primary_scores[idx],
        "is_correct": self.primary_scores[idx] >= 0.5,
    }


def get_all_generations(self) -> List[Dict[str, Any]]:
    """Get all generations for this task as a list of dicts.
    
    Useful for:
    - GRPO batch processing (all generations + scores)
    - Logging (includes full generation history)
    - Analysis (per-generation statistics)
    - Replay (reconstructing exact scenario)
    
    Returns:
        List of generation dicts, one per generation
    
    Examples:
        >>> batch = task.get_all_generations()
        >>> len(batch)
        4  # num_generations
        >>> [g["primary_score"] for g in batch]
        [1.0, 0.0, 0.0, 1.0]
        >>> [g["output"] for g in batch]
        ['<answer>42</answer>', '<answer>41</answer>', ...]
    """
    return [self.get_generation(i) for i in range(len(self.generation_outputs))]


def get_batch_stats(self) -> Dict[str, Any]:
    """Get statistics about the GRPO batch.
    
    Computes aggregate metrics: size, avg/min/max scores, correctness count.
    Empty batch returns batch_size=0.
    
    Returns:
        Dictionary with batch_size, primary_scores, avg_score, max_score, 
        min_score, num_correct
    
    Examples:
        >>> stats = task.get_batch_stats()
        >>> stats["batch_size"]
        4
        >>> stats["avg_score"]
        0.5
        >>> stats["num_correct"]
        2
    """
    if not self.primary_scores:
        return {
            "batch_size": 0,
            "primary_scores": [],
            "avg_score": 0.0,
            "max_score": 0.0,
            "min_score": 0.0,
            "num_correct": 0,
        }
    
    scores = self.primary_scores
    return {
        "batch_size": len(scores),
        "primary_scores": list(scores),
        "avg_score": sum(scores) / len(scores),
        "max_score": max(scores),
        "min_score": min(scores),
        "num_correct": sum(1 for s in scores if s >= 0.5),
    }
```

### Update to_dict() Method

```python
def to_dict(self) -> Dict[str, Any]:
    """Convert task to dictionary for logging.
    
    Includes both backward-compatible fields (latest generation)
    and new generation tracking fields (all generations).
    """
    return {
        # Existing fields
        "task_id": self.task_id,
        "task_name": self.task_name,
        "task_type": self.task_type,
        "level": self.level,
        "language": self.language,
        "reasoning_language": self.reasoning_language,
        "prompt": self.prompt,
        "judge_system_prompt": self.judge_system_prompt,
        "expected_answer": self.expected_answer,
        "model_output": self.model_output,  # Latest
        "created_at": self.created_at.isoformat() if self.created_at else None,
        "first_response_at": (
            self.first_response_at.isoformat() if self.first_response_at else None
        ),
        "is_correct": self.is_correct,  # Latest
        "task_rewards": [
            {
                "reward_function_name": r.reward_function_name,
                "score": r.score,
                "info": r.info,
            }
            for r in self.task_rewards
        ],  # Latest
        
        # NEW: Generation data (all)
        "generation_count": len(self.generation_outputs),
        "primary_scores": self.primary_scores,  # All
        "generation_data": [
            {
                "generation_idx": i,
                "output": self.generation_outputs[i],
                "primary_score": self.primary_scores[i],
                "is_correct": self.primary_scores[i] >= 0.5,
                "rewards": [
                    {
                        "reward_function_name": r.reward_function_name,
                        "score": r.score,
                        "info": r.info,
                    }
                    for r in self.generation_rewards[i]
                ],
            }
            for i in range(len(self.generation_outputs))
        ],
    }
```

---

## Phase 2: CurriculumLearning Changes

### Remove These Instance Variables

```python
# DELETE from __init__:
# self.grpo_batch_primary_scores: Dict[str, List[float]] = {}
# self.grpo_batch_outputs: Dict[str, List[str]] = {}
```

### Update compute_reward() Method

**Find this code:**
```python
def compute_reward(self, task: Task) -> float:
    # ... earlier code ...
    
    # Extract base task_id (remove instance counter)
    base_task_id = task_id.rsplit("_", 1)[0] if "_" in task_id else task_id
    if base_task_id not in self.grpo_batch_primary_scores:
        self.grpo_batch_primary_scores[base_task_id] = []
        self.grpo_batch_outputs[base_task_id] = []

    self.grpo_batch_primary_scores[base_task_id].append(score)
    self.grpo_batch_outputs[base_task_id].append(task.model_output)
```

**Replace with:**
```python
def compute_reward(self, task: Task) -> float:
    # ... earlier code ...
    
    # Add generation to task (replaces dict-based tracking)
    generation_idx = task.add_generation(
        output=task.model_output,
        rewards=task_rewards,
        primary_score=score
    )
```

### Update Batch Completion Logic

**Find this code:**
```python
# Check if we have a complete group (GRPO batch size)
if len(self.grpo_batch_primary_scores[base_task_id]) >= self.num_generations:
    # Complete group: track success at prompt level
    primary_scores = self.grpo_batch_primary_scores[base_task_id]
    if task.task_type != "truthy":
        self._track_success_group(task.level, primary_scores)

    # Increment step counter ONCE per complete prompt group
    self.global_step += 1

    # Check if we should advance/demote level
    self._update_level()

    # Log evaluation if configured (only when batch is complete)
    if self.log_file is not None:
        log_entry = task.to_dict()
        log_entry["timestamp"] = datetime.datetime.now().isoformat()
        log_entry["primary_score"] = primary_reward.score
        log_entry["aux_scores"] = {
            name: data["score"] for name, data in aux_score_dict.items()
        }
        log_entry["info"] = {
            "primary": primary_reward.info,
            **{
                f"aux_{name}": data["info"]
                for name, data in aux_score_dict.items()
            },
        }
        # Add GRPO batch information
        log_entry["grpo_batch_size"] = len(primary_scores)
        log_entry["grpo_primary_scores"] = primary_scores
        log_entry["grpo_model_outputs"] = self.grpo_batch_outputs[base_task_id]
        with open(self.log_file, "a", encoding="utf-8") as f:
            json.dump(log_entry, f, ensure_ascii=False)
            f.write("\n")

    # Clean up completed batch
    del self.grpo_batch_primary_scores[base_task_id]
    del self.grpo_batch_outputs[base_task_id]
```

**Replace with:**
```python
# Check if we have a complete group (GRPO batch size)
if len(task.generation_outputs) >= self.num_generations:
    # Complete group: track success at prompt level
    batch_data = task.get_all_generations()
    primary_scores = [g["primary_score"] for g in batch_data]
    
    if task.task_type != "truthy":
        self._track_success_group(task.level, primary_scores)

    # Increment step counter ONCE per complete prompt group
    self.global_step += 1

    # Check if we should advance/demote level
    self._update_level()

    # Log evaluation if configured (only when batch is complete)
    if self.log_file is not None:
        log_entry = task.to_dict()
        log_entry["timestamp"] = datetime.datetime.now().isoformat()
        log_entry["primary_score"] = primary_reward.score
        log_entry["aux_scores"] = {
            name: data["score"] for name, data in aux_score_dict.items()
        }
        log_entry["info"] = {
            "primary": primary_reward.info,
            **{
                f"aux_{name}": data["info"]
                for name, data in aux_score_dict.items()
            },
        }
        # Add GRPO batch information (from task, not dicts)
        log_entry["grpo_batch_size"] = len(batch_data)
        log_entry["grpo_primary_scores"] = primary_scores
        log_entry["grpo_model_outputs"] = [g["output"] for g in batch_data]
        
        with open(self.log_file, "a", encoding="utf-8") as f:
            json.dump(log_entry, f, ensure_ascii=False)
            f.write("\n")

    # No cleanup needed - Task owns the data, Session stores it
    # Data persists for analysis and logging
```

### Update Single Evaluation Mode

**Find this code:**
```python
elif self.num_generations == 1:
    # Single evaluation mode (non-GRPO): log immediately
    if task.task_type != "truthy":
        self._track_success_group(task.level, [score])
    self.global_step += 1
    self._update_level()

    # Log evaluation if configured
    if self.log_file is not None:
        log_entry = task.to_dict()
        # ... logging ...
```

**No changes needed** - `task.to_dict()` now includes generation data automatically.

---

## Phase 3: Session Enhancements

### Add New Methods to Session Class

```python
def get_batch_data(self, task_id: str) -> Optional[List[Dict[str, Any]]]:
    """Get all generations and their scores for a task.
    
    Returns the complete GRPO batch data if available.
    This enables post-training analysis of generation history.
    
    Args:
        task_id: Task identifier
    
    Returns:
        List of generation dicts (from task.get_all_generations())
        or None if task not found
    
    Examples:
        >>> batch = session.get_batch_data("task_123")
        >>> len(batch)
        4  # num_generations
        >>> batch[0]["primary_score"]
        1.0
        >>> batch[0]["output"]
        '<answer>42</answer>'
    """
    task = self.get_task(task_id)
    if task is None:
        return None
    return task.get_all_generations()


def get_batch_stats(self, task_id: str) -> Optional[Dict[str, Any]]:
    """Get statistics about a task's GRPO batch.
    
    Computes: batch_size, avg_score, min/max scores, num_correct
    
    Args:
        task_id: Task identifier
    
    Returns:
        Statistics dict or None if task not found
    
    Examples:
        >>> stats = session.get_batch_stats("task_123")
        >>> stats["avg_score"]
        0.5
        >>> stats["num_correct"]
        2
    """
    task = self.get_task(task_id)
    if task is None:
        return None
    return task.get_batch_stats()


def get_batch_data_for_all(self) -> Dict[str, List[Dict[str, Any]]]:
    """Get batch data for all tasks in session.
    
    Useful for comprehensive analysis of all generations.
    
    Returns:
        Dict mapping task_id → generation list
    
    Examples:
        >>> batches = session.get_batch_data_for_all()
        >>> len(batches["task_123"])
        4  # num_generations for task_123
    """
    return {
        task_id: task.get_all_generations()
        for task_id, task in self.tasks.items()
        if task.generation_outputs
    }
```

---

## Testing Examples

### Unit Tests for Task

```python
def test_task_add_generation():
    """Test adding generations to a task."""
    task = Task(
        task_id="test_1",
        task_name="Math Problem",
        task_type="math",
        level=0,
        prompt="What is 40+2?",
        expected_answer="42"
    )
    
    rewards_1 = [RewardFunctionScore(1.0, "primary", "Correct")]
    idx1 = task.add_generation(
        output="<answer>42</answer>",
        rewards=rewards_1,
        primary_score=1.0
    )
    assert idx1 == 0
    assert task.model_output == "<answer>42</answer>"
    assert task.is_correct == True
    
    rewards_2 = [RewardFunctionScore(0.0, "primary", "Wrong")]
    idx2 = task.add_generation(
        output="<answer>41</answer>",
        rewards=rewards_2,
        primary_score=0.0
    )
    assert idx2 == 1
    assert task.model_output == "<answer>41</answer>"  # Updated
    assert task.is_correct == False  # Updated


def test_task_get_generation():
    """Test retrieving a specific generation."""
    task = Task(...)
    task.add_generation("<answer>42</answer>", [...], 1.0)
    task.add_generation("<answer>41</answer>", [...], 0.0)
    
    gen0 = task.get_generation(0)
    assert gen0["generation_idx"] == 0
    assert gen0["output"] == "<answer>42</answer>"
    assert gen0["primary_score"] == 1.0
    assert gen0["is_correct"] == True
    
    gen1 = task.get_generation(1)
    assert gen1["generation_idx"] == 1
    assert gen1["output"] == "<answer>41</answer>"
    assert gen1["primary_score"] == 0.0
    assert gen1["is_correct"] == False


def test_task_get_all_generations():
    """Test getting all generations."""
    task = Task(...)
    task.add_generation("out1", [...], 1.0)
    task.add_generation("out2", [...], 0.0)
    task.add_generation("out3", [...], 1.0)
    
    batch = task.get_all_generations()
    assert len(batch) == 3
    assert batch[0]["primary_score"] == 1.0
    assert batch[1]["primary_score"] == 0.0
    assert batch[2]["primary_score"] == 1.0


def test_task_get_batch_stats():
    """Test batch statistics."""
    task = Task(...)
    task.add_generation("out1", [...], 1.0)
    task.add_generation("out2", [...], 0.0)
    task.add_generation("out3", [...], 1.0)
    task.add_generation("out4", [...], 0.0)
    
    stats = task.get_batch_stats()
    assert stats["batch_size"] == 4
    assert stats["avg_score"] == 0.5
    assert stats["max_score"] == 1.0
    assert stats["min_score"] == 0.0
    assert stats["num_correct"] == 2
```

### Integration Tests for CurriculumLearning

```python
def test_curriculum_grpo_batch_in_task():
    """Test that GRPO batch data is stored in task."""
    curriculum = CurriculumLearning(num_generations=2)
    task = Task(...)
    
    # First generation
    curriculum.compute_reward(task)
    assert len(task.generation_outputs) == 1
    assert len(task.primary_scores) == 1
    
    # Second generation (completes batch)
    curriculum.compute_reward(task)
    assert len(task.generation_outputs) == 2
    assert len(task.primary_scores) == 2
    
    # Verify data
    batch = task.get_all_generations()
    assert len(batch) == 2
    for i, gen in enumerate(batch):
        assert "generation_idx" in gen
        assert "output" in gen
        assert "primary_score" in gen
        assert "rewards" in gen


def test_session_get_batch_data():
    """Test querying batch data from session."""
    session = Session()
    task = Task(...)
    task.add_generation("out1", [...], 1.0)
    task.add_generation("out2", [...], 0.0)
    session.add_task(task)
    
    batch = session.get_batch_data(task.task_id)
    assert batch is not None
    assert len(batch) == 2
    assert batch[0]["primary_score"] == 1.0


def test_session_get_batch_stats():
    """Test batch statistics from session."""
    session = Session()
    task = Task(...)
    task.add_generation("out1", [...], 1.0)
    task.add_generation("out2", [...], 0.0)
    session.add_task(task)
    
    stats = session.get_batch_stats(task.task_id)
    assert stats["batch_size"] == 2
    assert stats["avg_score"] == 0.5
    assert stats["num_correct"] == 1
```

---

## Migration Checklist

### Phase 1: Task Class
- [ ] Add `generation_outputs`, `generation_rewards`, `primary_scores` fields
- [ ] Implement `add_generation()` method
- [ ] Implement `get_generation()` method
- [ ] Implement `get_all_generations()` method
- [ ] Implement `get_batch_stats()` method
- [ ] Update `to_dict()` method
- [ ] Add docstrings to all new methods
- [ ] Add unit tests for new methods
- [ ] Run full test suite - **All tests should pass**
- [ ] Verify backward compat: `task.model_output` and `task.task_rewards` work

### Phase 2: CurriculumLearning
- [ ] Remove `self.grpo_batch_primary_scores` instance variable
- [ ] Remove `self.grpo_batch_outputs` instance variable
- [ ] Update `compute_reward()` to call `task.add_generation()`
- [ ] Update batch completion logic to use `task.get_all_generations()`
- [ ] Remove batch cleanup code (`del self.grpo_batch_*`)
- [ ] Update logging to use generation data from task
- [ ] Update docstrings
- [ ] Run curriculum tests - **All tests should pass**
- [ ] Verify no dict dereference errors in logs
- [ ] Check JSONL format includes all generation data

### Phase 3: Session
- [ ] Add `get_batch_data()` method
- [ ] Add `get_batch_stats()` method
- [ ] Add `get_batch_data_for_all()` method
- [ ] Add docstrings
- [ ] Add unit tests for new methods
- [ ] Run session tests - **All tests should pass**

---

## Verification Points

After each phase, verify:
1. ✅ All tests pass (`pytest tests/`)
2. ✅ No import errors
3. ✅ No AttributeErrors in curriculum
4. ✅ JSONL logs are valid JSON
5. ✅ Generation data is complete in logs
6. ✅ Curriculum progression unaffected
7. ✅ No performance regression
8. ✅ Backward compat APIs work
