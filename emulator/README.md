# Training Simulator Emulator

A comprehensive toolkit for simulating curriculum learning progression with synthetic response combinations.

## Contents

- **training_simulator.py**: Core simulator with `TrainingSimulator` class for running training scenarios
- **advanced_scenarios.py**: Pre-built complex scenarios and `ResponsePattern` shortcuts
- **examples.py**: 10+ ready-to-run examples demonstrating different scenarios
- **TRAINING_SIMULATOR_GUIDE.md**: Complete user documentation with examples and interpretation guide

## Quick Start

```python
from emulator import TrainingSimulator, ResponsePattern

# Create simulator
simulator = TrainingSimulator(num_generations=4)

# Run a simple scenario
configs = [ResponsePattern.perfect()] * 50
simulator.run_scenario("Perfect Responses", response_configs=configs)
simulator.print_results()
```

## Running Examples

```bash
# View all response patterns
python -m emulator.examples patterns

# Run example 1 (Perfect Responses)
python -m emulator.examples 1

# Run example 3 (Gradual Improvement)
python -m emulator.examples 3

# Run example 11 - COMPREHENSIVE REWARD FUNCTION TEST
# Tests all reward functions with up-down-up scenario
python -m emulator.examples 11

# Run all advanced scenarios
python -m emulator.examples 7
```

## Files Overview

### training_simulator.py
- `RewardSnapshot`: Dataclass for tracking metrics at each step
- `TrainingSimulator`: Main class with methods:
  - `generate_response()`: Create synthetic responses with specified properties
  - `run_scenario()`: Execute training scenario with batched processing
  - `print_results()`: Format output for analysis
  - `save_results()`: Export to JSON

### advanced_scenarios.py
- `ResponsePattern`: 8 predefined response patterns (perfect, format_only, no_think, etc.)
- `AdvancedScenarios`: 5 complex training scenarios
  - mixed_quality_progression: Realistic 4-phase learning
  - format_first_then_correctness: Two-phase progression
  - difficulty_mismatch: Task too hard scenario
  - recovery_from_collapse: Collapse and recovery
  - cascade_success: Natural progression

### examples.py
10 standalone examples covering:
1. Perfect responses baseline
2. Format issues (missing think tags)
3. Gradual improvement over time
4. Different format error types
5. Correctness vs format impact
6. Collapse and recovery pattern
7. Using advanced scenarios
8. Custom patterns with randomness
9. Comparing multiple scenarios
10. Saving and loading results

## Response Configuration Format

Responses are configured as 3-tuples:
```python
(has_think_format: bool, has_answer_format: bool, is_correct: bool)
```

Example usage:
```python
# Perfect response: both tags present, answer is correct
config = (True, True, True)

# Format only: both tags present, answer is wrong
config = (True, True, False)

# Missing think tag: only answer tag, answer is correct
config = (False, True, True)

# All bad: no tags, answer is wrong
config = (False, False, False)
```

## Documentation

See [TRAINING_SIMULATOR_GUIDE.md](TRAINING_SIMULATOR_GUIDE.md) for:
- Detailed configuration options
- Metrics interpretation guide
- 15+ example scenarios
- Custom scenario creation
- Common use cases

## Integration with Curriculum Learning

The simulator integrates with the main curriculum system to:
- Track level progression based on success rates
- Apply strict gates (format AND correctness required)
- Simulate GRPO batch-based training
- Record detailed metrics for analysis

Level advancement uses:
- Success rate threshold: 80% (configurable)
- Variance threshold: 0.05 (configurable for stability)
- Per-task-type windows for independent progression

## Comprehensive Reward Function Testing

### test_all_rewards.py - Example 11

Tests all reward functions simultaneously with a realistic up-down-up scenario:

**What it tests:**
- **Phase 1** (26 steps): Perfect responses (both format + correctness)
- **Phase 2** (25 steps): All bad responses (no format, wrong answer)
- **Phase 3** (25 steps): Recovery (both format + correctness again)

**Reward Functions Enabled:**
- ✓ `use_format`: Format validation (XML-like tags)
- ✓ `use_reasoning_steps`: Chain-of-thought reasoning bonus
- ✓ `use_lang_consistency`: Language consistency penalty
- ✓ `use_repetition`: Repetition phrase penalty
- ✓ `use_length`: Response length regularization

**What it verifies:**
1. All auxiliary reward functions are initialized and available
2. Primary scores (correctness) drive level progression
3. Auxiliary rewards don't override primary failures
4. Format validation works as a hard gate
5. Success rate tracking respects sliding windows
6. Level changes happen at correct thresholds

**Output includes:**
- Phase-by-phase success rates and level changes
- Detailed step-by-step progression
- Response type impact analysis
- Reward function activation status
- JSON results file for downstream analysis

**Run it:**
```bash
python -m emulator.examples 11
```

**Example output:**
```
COMPREHENSIVE REWARD FUNCTION TEST
Scenario: UP → DOWN → UP (Good → Bad → Recovery)
================================================================================

Reward Functions Configuration:
  ✓ use_format: True
  ✓ use_reasoning_steps: True
  ✓ use_lang_consistency: True
  ✓ use_repetition: True
  ✓ use_length: True
  ✓ aux_weight: 0.2
  ✓ num_generations: 4

PHASE 1: GOOD RESPONSES (Both Format + Correctness)
────────────────────────────────────────────────────
  Final Level: 2
  Final Success Rate: 100.0%
  Total Steps: 7

PHASE 2: BAD RESPONSES (No Format + Wrong Answer)
────────────────────────────────────────────────────
  Final Level: 0
  Final Success Rate: 0.0%
  Total Steps: 7

PHASE 3: RECOVERY (Both Format + Correctness)
────────────────────────────────────────────────────
  Final Level: 1
  Final Success Rate: 100.0%
  Total Steps: 7

REWARD FUNCTION ANALYSIS
================================================================================

Phase Summary:
────────────────────────────────────────────────────────────────────────────────
Phase                     Level    Success Rate          Steps
────────────────────────────────────────────────────────────────────────────────
Phase 1: Good             2        100.0%                7
Phase 2: Bad              0        0.0%                  7
Phase 3: Recovery         1        100.0%                7
```

This test confirms that the curriculum learning system correctly:
- Advances levels when success rate increases (Phase 1 → Phase 3)
- Demotes levels when performance degrades (Phase 2)
- Uses all available reward functions
- Applies proper gating for format validation
