# Reward Function Unit Tests

This document describes the unit tests for the reward functions in the `infinite_rl` package.

## Overview

A comprehensive test suite for current reward function types:
- **Coding** - Multi-language code execution and output validation (Python, JavaScript, TypeScript)
- **Math** - Symbolic mathematics with SymPy

## Test Coverage

### 1. CodingRewardFunction (3 tests)
Tests the code execution and validation:
- ✅ `test_valid_python_code_with_json_output` - Correct code with matching output
- ✅ `test_missing_code_block` - Handles missing code blocks gracefully
- ✅ `test_syntax_error_in_code` - Handles syntax errors (stderr)

### 2. MathRewardFunction (5 tests)
Tests symbolic math evaluation:
- ✅ `test_correct_integral_answer` - Correct symbolic math evaluation
- ✅ `test_missing_answer_tag` - Handles missing answer tags
- ✅ `test_incorrect_mathematical_answer` - Detects incorrect equations
- ✅ `test_integer_expected_output` - Supports integer comparison
- ✅ `test_integer_with_wrong_value` - Similarity-based scoring for integers



## Running the Tests

```bash
# Run all tests
python -m pytest tests/test_reward_functions.py -v

# Run specific test class
python -m pytest tests/test_reward_functions.py::TestCodingRewardFunction -v

# Run with coverage
python -m pytest tests/test_reward_functions.py --cov=infinite_rl.reward_functions
```

## Key Design Patterns

### 1. Reward Function Interface
All reward functions follow the same interface:
```python
def compute_reward(
    self, 
    model_output: str, 
    expected_output: Union[str, int, Callable]
) -> RewardFunctionScore:
```

Returns `RewardFunctionScore(format_score: float, correctness_score: float)`

### 2. Expected Output Types
Different reward functions support different input types:
- **String**: Direct comparison or symbolic evaluation
- **Integer**: Numeric comparison with threshold-based scoring
- **Callable**: Custom validator function

### 4. Binary Correctness Scoring
Correctness scores follow a binary pattern for deterministic evaluation:
- When similarity/matching > 0.5 threshold → 1.0
- When similarity/matching ≤ 0.5 threshold → 0.0

## Test Results

```
15 passed in 4.57s
```

All tests pass successfully, validating:
✅ Format score computation (code blocks, tags, syntax)
✅ Correctness score computation (symbolic math, CSS selectors, similarity)
✅ Error handling (missing blocks, invalid syntax, exceptions)
✅ Support for multiple expected output types
✅ Callable validator function support
