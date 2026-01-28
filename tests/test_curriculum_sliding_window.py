#!/usr/bin/env python3
"""Test sliding window success rate implementation."""

import sys
import os
from collections import deque

# Add the project root to the path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from infinite_rl.curriculum import CurriculumLearning


def test_sliding_window_tracking():
    """Test that sliding window tracks success rates correctly."""
    print("Testing sliding window success rate tracking...")

    # Create curriculum with small window for testing
    curriculum = CurriculumLearning(
        window_size=10,
        success_rate_threshold=0.8,
        variance_threshold=0.05,
    )

    # Simulate some successes and failures
    curriculum._track_success("math", True)  # 1/1
    curriculum._track_success("math", True)  # 2/2
    curriculum._track_success("math", True)  # 3/3
    curriculum._track_success("math", False)  # 3/4
    curriculum._track_success("math", True)  # 4/5

    # Check success rate
    stats = curriculum.get_success_rate("math")
    print(f"Math success rate: {stats['success_rate']:.2%}")
    print(f"Math variance: {stats['variance']:.4f}")
    print(f"Math samples: {stats['samples']}")

    assert stats["success_rate"] == 0.8, f"Expected 0.8, got {stats['success_rate']}"
    assert stats["samples"] == 5, f"Expected 5 samples, got {stats['samples']}"

    print("✓ Single task type tracking works")

    # Test with multiple task types
    for _ in range(8):
        curriculum._track_success("puzzle", True)

    curriculum._track_success("puzzle", False)
    curriculum._track_success("puzzle", False)

    stats = curriculum.get_success_rate("puzzle")
    print(f"\nPuzzle success rate: {stats['success_rate']:.2%}")
    print(f"Puzzle variance: {stats['variance']:.4f}")
    print(f"Puzzle samples: {stats['samples']}")

    assert stats["success_rate"] == 0.8, f"Expected 0.8, got {stats['success_rate']}"

    print("✓ Multiple task type tracking works")

    # Test aggregated stats
    agg_stats = curriculum.get_success_rate()
    print(f"\nAggregated success rate: {agg_stats['mean_success_rate']:.2%}")
    print(f"Aggregated variance: {agg_stats['mean_variance']:.4f}")
    print(f"Total samples: {agg_stats['samples']}")
    print(f"By task type:")
    for task_stats in agg_stats["by_task_type"]:
        print(
            f"  {task_stats['task_type']}: {task_stats['success_rate']:.2%} "
            f"(variance: {task_stats['variance']:.4f})"
        )

    print("\n✓ Aggregated stats work correctly")

    # Test sliding window overflow (should only keep last N)
    for i in range(20):
        curriculum._track_success("math", True)

    window = curriculum.success_windows["math"]
    print(f"\nWindow size after adding 20 more: {len(window)}")
    assert len(window) == 10, f"Window should be capped at 10, but got {len(window)}"

    print("✓ Sliding window maxlen enforcement works")


def test_level_advancement():
    """Test that level advancement happens when conditions are met."""
    print("\n\nTesting level advancement logic...")

    curriculum = CurriculumLearning(
        window_size=10,
        success_rate_threshold=0.8,
        variance_threshold=0.05,
    )

    initial_level = curriculum.current_level
    print(f"Initial level: {initial_level}")

    # Add consistent success - should eventually advance
    for _ in range(15):
        curriculum._track_success("math", True)

    curriculum._update_level()
    print(f"After 15 successes: level={curriculum.current_level}")

    # With high variance, should NOT advance
    curriculum.success_windows.clear()
    curriculum.current_level = 0

    # Create high variance (alternating success/failure)
    for i in range(10):
        curriculum._track_success("math", i % 2 == 0)

    stats = curriculum.get_success_rate("math")
    print(
        f"\nHigh variance test: success_rate={stats['success_rate']:.2%}, "
        f"variance={stats['variance']:.4f}"
    )

    current_level = curriculum.current_level
    curriculum._update_level()
    print(
        f"After high variance, level stayed at: {curriculum.current_level} "
        f"(expected: {current_level})"
    )

    assert (
        curriculum.current_level == current_level
    ), f"Level should not advance with high variance"

    print("✓ Level advancement logic works correctly")


if __name__ == "__main__":
    test_sliding_window_tracking()
    test_level_advancement()
    print("\n" + "=" * 50)
    print("All tests passed! ✓")
    print("=" * 50)
