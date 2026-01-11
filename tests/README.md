# Reward Function Unit Tests

This document describes the unit tests for the reward functions in the `infinite_rl` package.

## Overview

A comprehensive test suite for all reward function types:
- **Coding** - Multi-language code execution and output validation
- **Math** - Symbolic mathematics with SymPy
- **HTML** - HTML syntax validation and CSS selector matching
- **Summarization** - Semantic similarity with mocked embeddings

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

### 3. HtmlRewardFunction (3 tests)
Tests HTML parsing and CSS selector validation:
- ✅ `test_valid_html_with_all_selectors` - Validates all CSS selectors match
- ✅ `test_missing_html_code_block` - Parses raw HTML without code blocks
- ✅ `test_missing_required_selectors` - Detects missing selectors

### 4. SummarizationRewardFunction (4 tests)
Tests semantic similarity with mocked SentenceTransformer:
- ✅ `test_valid_summary_with_high_similarity` - High similarity (0.95) = 1.0 score
- ✅ `test_valid_summary_with_low_similarity` - Low similarity (0.3) = 0.0 score
- ✅ `test_missing_summary_tag` - Handles missing summary tags
- ✅ `test_summary_with_callable_validator` - Supports custom validator functions

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

### 1. Mocking SentenceTransformer
The summarization tests use `@patch()` to mock the expensive `SentenceTransformer` import:

```python
@patch("infinite_rl.reward_functions.summarization.SentenceTransformer")
def test_valid_summary_with_high_similarity(self, mock_transformer_class):
    mock_model_instance = MagicMock()
    mock_transformer_class.return_value = mock_model_instance
    # ... set up embeddings and similarity mocks
```

### 2. Reward Function Interface
All reward functions follow the same interface:
```python
def compute_reward(
    self, 
    model_output: str, 
    expected_output: Union[str, int, Callable]
) -> RewardFunctionScore:
```

Returns `RewardFunctionScore(format_score: float, correctness_score: float)`

### 3. Expected Output Types
Different reward functions support different input types:
- **String**: Direct comparison or symbolic evaluation
- **Integer**: Numeric comparison with threshold-based scoring
- **Dict**: For HTML (selectors list), Summarization (JSON structure)
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
