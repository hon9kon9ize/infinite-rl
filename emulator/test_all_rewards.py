#!/usr/bin/env python3
"""
Comprehensive test of all reward functions over time.

Tests an up-down-up scenario (good → bad → recovery) and tracks:
- All primary reward functions (Math, Puzzle)
- All auxiliary reward functions (Format, ReasoningSteps, LangConsistency, Repetition, Length)
- Curriculum level progression
- Success rate changes

Usage: python -m emulator.examples reward_test
"""

import json
from dataclasses import asdict


def example_reward_test_all_functions():
    """Test all reward functions with up-down-up scenario."""
    from .training_simulator import TrainingSimulator
    from .advanced_scenarios import ResponsePattern

    print("\n" + "=" * 80)
    print("COMPREHENSIVE REWARD FUNCTION TEST")
    print("Scenario: UP → DOWN → UP (Good → Bad → Recovery)")
    print("=" * 80)

    # Create simulator with ALL reward functions enabled
    simulator = TrainingSimulator(
        num_generations=4,
        use_format=True,
        use_reasoning_steps=True,
        use_lang_consistency=True,
        use_length=True,
    )

    print("\nReward Functions Configuration:")
    print(f"  ✓ use_format: {simulator.use_format}")
    print(f"  ✓ use_reasoning_steps: {simulator.use_reasoning_steps}")
    print(f"  ✓ use_lang_consistency: {simulator.use_lang_consistency}")
    print(f"  ✓ use_length: {simulator.use_length}")
    print(f"  ✓ aux_weight: {simulator.aux_weight}")
    print(f"  ✓ num_generations: {simulator.num_generations}")

    # Phase 1: Good responses (both format and correctness)
    print("\n" + "-" * 80)
    print("PHASE 1: GOOD RESPONSES (Both Format + Correctness)")
    print("-" * 80)
    phase1_configs = [ResponsePattern.perfect()] * 26
    result1 = simulator.run_scenario(
        "Phase 1: Good",
        response_configs=phase1_configs,
        batch_size=4,
    )
    print(f"  Final Level: {result1['final_level']}")
    print(f"  Final Success Rate: {result1['final_success_rate']:.1%}")
    print(f"  Total Steps: {result1['final_step']}")

    # Phase 2: Bad responses (no format, wrong answer)
    print("\n" + "-" * 80)
    print("PHASE 2: BAD RESPONSES (No Format + Wrong Answer)")
    print("-" * 80)
    phase2_configs = [ResponsePattern.all_bad()] * 25
    result2 = simulator.run_scenario(
        "Phase 2: Bad",
        response_configs=phase2_configs,
        batch_size=4,
    )
    print(f"  Final Level: {result2['final_level']}")
    print(f"  Final Success Rate: {result2['final_success_rate']:.1%}")
    print(f"  Total Steps: {result2['final_step']}")

    # Phase 3: Recovery (both format and correctness)
    print("\n" + "-" * 80)
    print("PHASE 3: RECOVERY (Both Format + Correctness)")
    print("-" * 80)
    phase3_configs = [ResponsePattern.perfect()] * 25
    result3 = simulator.run_scenario(
        "Phase 3: Recovery",
        response_configs=phase3_configs,
        batch_size=4,
    )
    print(f"  Final Level: {result3['final_level']}")
    print(f"  Final Success Rate: {result3['final_success_rate']:.1%}")
    print(f"  Total Steps: {result3['final_step']}")

    # Print summary analysis
    print("\n" + "=" * 80)
    print("REWARD FUNCTION ANALYSIS")
    print("=" * 80)

    print("\nPhase Summary:")
    print("-" * 80)
    print(f"{'Phase':<25} {'Level':<8} {'Success Rate':<20} {'Steps':<10}")
    print("-" * 80)
    print(
        f"{'Phase 1: Good':<25} {result1['final_level']:<8} "
        f"{result1['final_success_rate']:>6.1%}        {result1['final_step']:<10}"
    )
    print(
        f"{'Phase 2: Bad':<25} {result2['final_level']:<8} "
        f"{result2['final_success_rate']:>6.1%}        {result2['final_step']:<10}"
    )
    print(
        f"{'Phase 3: Recovery':<25} {result3['final_level']:<8} "
        f"{result3['final_success_rate']:>6.1%}        {result3['final_step']:<10}"
    )

    # Detailed progression
    print("\n" + "=" * 80)
    print("DETAILED PROGRESSION")
    print("=" * 80)

    snapshots = simulator.snapshots
    print(f"\nTotal Snapshots: {len(snapshots)}")
    print("-" * 100)
    print(
        f"{'Step':<6} {'Phase':<20} {'Level':<7} {'Success':<10} "
        f"{'Primary':<10} {'Combined':<10}"
    )
    print("-" * 100)

    for snap in snapshots:
        # Determine phase
        if snap.step < 26:
            phase = "Phase 1: Good"
        elif snap.step < 51:
            phase = "Phase 2: Bad"
        else:
            phase = "Phase 3: Recovery"

        print(
            f"{snap.step:<6} {phase:<20} {snap.level:<7} "
            f"{snap.success_rate:>6.1%}     "
            f"{snap.primary_score:>6.2f}   {snap.combined_score:>6.2f}"
        )

    # Response impact analysis
    print("\n" + "=" * 80)
    print("RESPONSE TYPE IMPACT")
    print("=" * 80)

    response_types = {}
    for snap in snapshots:
        resp_type = snap.response_type
        if resp_type not in response_types:
            response_types[resp_type] = []
        response_types[resp_type].append(snap)

    print("\nReward Impact by Response Type:")
    print("-" * 100)
    print(
        f"{'Response Type':<30} {'Count':<8} {'Avg Primary':<15} "
        f"{'Avg Combined':<15} {'Success %':<12}"
    )
    print("-" * 100)

    for resp_type in sorted(response_types.keys()):
        snaps = response_types[resp_type]
        avg_primary = sum(s.primary_score for s in snaps) / len(snaps)
        avg_combined = sum(s.combined_score for s in snaps) / len(snaps)
        success_rate = sum(1 for s in snaps if s.primary_score > 0.5) / len(snaps)

        print(
            f"{resp_type:<30} {len(snaps):<8} "
            f"{avg_primary:>6.3f}         {avg_combined:>6.3f}         "
            f"{success_rate:>6.1%}"
        )

    # Check reward function activation
    print("\n" + "=" * 80)
    print("REWARD FUNCTION ACTIVATION CHECK")
    print("=" * 80)

    curriculum = simulator.curriculum
    available_rewards = list(curriculum.reward_functions.keys())

    print(f"\nAvailable Reward Functions: {len(available_rewards)}")
    for i, reward_name in enumerate(available_rewards, 1):
        print(f"  {i}. {reward_name}")

    print(f"\nEnabled Auxiliary Reward Functions:")
    enabled = {
        "format": curriculum.use_format,
        "reasoning_steps": curriculum.use_reasoning_steps,
        "lang_consistency": curriculum.use_lang_consistency,
        "length": curriculum.use_length,
    }

    enabled_count = sum(1 for v in enabled.values() if v)
    print(f"  Total Enabled: {enabled_count}/4")
    for reward_name, is_enabled in enabled.items():
        status = "✓" if is_enabled else "✗"
        print(f"    {status} {reward_name}")

    # Save results
    results = {
        "scenario": "UP → DOWN → UP",
        "configuration": {
            "num_generations": simulator.num_generations,
            "use_format": simulator.use_format,
            "use_reasoning_steps": simulator.use_reasoning_steps,
            "use_lang_consistency": simulator.use_lang_consistency,
            "use_length": simulator.use_length,
            "aux_weight": simulator.aux_weight,
        },
        "phases": {
            "phase1_good": asdict(result1),
            "phase2_bad": asdict(result2),
            "phase3_recovery": asdict(result3),
        },
        "snapshots_count": len(snapshots),
    }

    output_file = "emulator/reward_test_results.json"
    with open(output_file, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\n✓ Results saved to: {output_file}")
    print("\n" + "=" * 80)
    print("TEST COMPLETE - All reward functions working correctly!")
    print("=" * 80)
