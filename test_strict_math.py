#!/usr/bin/env python3
"""Test strict math validation."""

from infinite_rl.reward_functions.math import _extract_number

# Valid cases (should pass)
valid_cases = [
    ("123", 123.0),
    ("42", 42.0),
    ("3.14", 3.14),
    ("-5", -5.0),
    ("+10", 10.0),
    ("1/2", 0.5),
    ("3 / 4", 0.75),
    ("1.5e-3", 0.0015),
    ("$1000", 1000.0),
    ("1,234", 1234.0),
    ("  42  ", 42.0),  # with whitespace
]

# Invalid cases (should fail)
invalid_cases = [
    "[123]",
    "<123>",
    "{123}",
    "[42]",
    "<answer>123</answer>",
    "123abc",
    "abc123",
    "The answer is 123",
    "123 units",
]

print("=== Testing Valid Cases ===")
for text, expected in valid_cases:
    result = _extract_number(text)
    status = "✓" if result == expected else "✗"
    print(f"{status} '{text}' -> {result} (expected {expected})")

print("\n=== Testing Invalid Cases (should all be None) ===")
for text in invalid_cases:
    result = _extract_number(text)
    status = "✓" if result is None else "✗"
    print(f"{status} '{text}' -> {result} (expected None)")
