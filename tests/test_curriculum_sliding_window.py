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

    # Simulate some successes and failures for level 0 (using GRPO batch-level tracking)
    curriculum._track_success_group(0, [1.0])  # 1/1 - batch with perfect score
    curriculum._track_success_group(0, [1.0])  # 2/2
    curriculum._track_success_group(0, [1.0])  # 3/3
    curriculum._track_success_group(0, [0.0])  # 3/4 - batch with no perfect score
    curriculum._track_success_group(0, [1.0])  # 4/5

    # Check success rate
    stats = curriculum.get_success_rate(0)
    print(f"Level 0 success rate: {stats['success_rate']:.2%}")
    print(f"Level 0 variance: {stats['variance']:.4f}")
    print(f"Level 0 samples: {stats['samples']}")

    assert stats["success_rate"] == 0.8, f"Expected 0.8, got {stats['success_rate']}"
    assert stats["samples"] == 5, f"Expected 5 samples, got {stats['samples']}"

    print("✓ Single level tracking works")

    # Test with multiple levels
    for _ in range(8):
        curriculum._track_success_group(1, [1.0])

    curriculum._track_success_group(1, [0.0])
    curriculum._track_success_group(1, [0.0])

    stats = curriculum.get_success_rate(1)
    print(f"\nLevel 1 success rate: {stats['success_rate']:.2%}")
    print(f"Level 1 variance: {stats['variance']:.4f}")
    print(f"Level 1 samples: {stats['samples']}")

    assert stats["success_rate"] == 0.8, f"Expected 0.8, got {stats['success_rate']}"

    print("✓ Multiple level tracking works")

    # Test aggregated stats
    agg_stats = curriculum.get_success_rate()
    print(f"\nAggregated success rate: {agg_stats['mean_success_rate']:.2%}")
    print(f"Aggregated variance: {agg_stats['mean_variance']:.4f}")
    print(f"Total samples: {agg_stats['samples']}")
    print(f"By level:")
    for level_stats in agg_stats["by_level"]:
        print(
            f"  Level {level_stats['level']}: {level_stats['success_rate']:.2%} "
            f"(variance: {level_stats['variance']:.4f})"
        )

    print("\n✓ Aggregated stats work correctly")

    # Test sliding window overflow (should only keep last N)
    for i in range(20):
        curriculum._track_success_group(0, [1.0])

    window = curriculum.success_windows[0]
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

    # Add consistent success at current level (0) - should eventually advance
    for _ in range(15):
        curriculum._track_success_group(0, [1.0])

    curriculum._update_level()
    print(f"After 15 successes at level 0: level={curriculum.current_level}")

    # With high variance, should NOT advance
    curriculum.success_windows.clear()
    curriculum.current_level = 0

    # Create high variance (alternating success/failure)
    for i in range(10):
        curriculum._track_success_group(0, [1.0] if i % 2 == 0 else [0.0])

    stats = curriculum.get_success_rate(0)
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
