#!/usr/bin/env python3
"""
Quick reference examples for training simulator.

Copy and paste any example to use immediately.
"""

# ============================================================================
# EXAMPLE 1: Simple Perfect Responses Scenario
# ============================================================================


def example_perfect():
    """All responses have correct format and correct answers."""
    from .training_simulator import TrainingSimulator

    simulator = TrainingSimulator(num_generations=4, use_format=True)
    simulator.run_scenario(
        "Perfect Responses",
        response_configs=[(True, True, True)] * 50,
    )
    simulator.print_results()


# ============================================================================
# EXAMPLE 2: Format Issues Scenario
# ============================================================================


def example_format_issues():
    """Responses struggle with proper formatting."""
    from .training_simulator import TrainingSimulator

    simulator = TrainingSimulator(num_generations=4, use_format=True)

    # 60% missing think tag, but answers are correct
    configs = [(False, True, True)] * 30 + [(True, True, True)] * 20

    simulator.run_scenario(
        "Format Issues (No Think Tag)",
        response_configs=configs,
    )
    simulator.print_results()


# ============================================================================
# EXAMPLE 3: Gradual Improvement
# ============================================================================


def example_gradual_improvement():
    """Model starts bad, gradually gets better."""
    from .training_simulator import TrainingSimulator
    from .advanced_scenarios import ResponsePattern

    simulator = TrainingSimulator(num_generations=4)

    configs = (
        [ResponsePattern.all_bad()] * 10  # Very bad start
        + [ResponsePattern.format_only()] * 15  # Learning format
        + [ResponsePattern.format_only()] * 15  # Format inconsistent
        + [ResponsePattern.perfect()] * 20  # Recovery
    )

    simulator.run_scenario(
        "Gradual Improvement",
        response_configs=configs,
    )
    simulator.print_results()


# ============================================================================
# EXAMPLE 4: Different Format Errors
# ============================================================================


def example_format_errors():
    """Compare different types of format errors."""
    from .training_simulator import TrainingSimulator
    from .advanced_scenarios import ResponsePattern

    simulator = TrainingSimulator(num_generations=4)

    configs = (
        [ResponsePattern.perfect()] * 10  # Baseline
        + [ResponsePattern.no_think()] * 10  # Missing think tag
        + [ResponsePattern.no_answer()] * 10  # Missing answer tag
        + [ResponsePattern.incomplete_format()] * 10  # Both missing
        + [ResponsePattern.perfect()] * 10  # Recovery
    )

    simulator.run_scenario(
        "Format Error Comparison",
        response_configs=configs,
    )
    simulator.print_results()


# ============================================================================
# EXAMPLE 5: Correctness vs Format
# ============================================================================


def example_correctness_vs_format():
    """Show impact of correctness when format is valid."""
    from .training_simulator import TrainingSimulator

    simulator = TrainingSimulator(num_generations=4)

    configs = []
    for i in range(100):
        if i < 30:
            # Always correct with valid format
            configs.append((True, True, True))
        elif i < 60:
            # Correct format but sometimes wrong answers
            configs.append((True, True, i % 3 != 0))
        else:
            # Good format AND answers
            configs.append((True, True, True))

    simulator.run_scenario(
        "Correctness vs Format Impact",
        response_configs=configs,
    )
    simulator.print_results()


# ============================================================================
# EXAMPLE 6: Collapse and Recovery
# ============================================================================


def example_collapse_recovery():
    """Model does well, then fails, then recovers."""
    from .training_simulator import TrainingSimulator
    from .advanced_scenarios import ResponsePattern

    simulator = TrainingSimulator(num_generations=4)

    configs = (
        [ResponsePattern.perfect()] * 20  # Good start
        + [ResponsePattern.all_bad()] * 20  # Collapse
        + [ResponsePattern.format_only()] * 10  # Struggling recovery
        + [ResponsePattern.perfect()] * 30  # Full recovery
    )

    simulator.run_scenario(
        "Collapse and Recovery",
        response_configs=configs,
    )
    simulator.print_results()


# ============================================================================
# EXAMPLE 7: Using Advanced Scenarios
# ============================================================================


def example_advanced_scenarios():
    """Use predefined advanced scenarios."""
    from .advanced_scenarios import (
        AdvancedScenarios,
        run_advanced_scenario,
    )

    scenarios = [
        ("Mixed Quality", AdvancedScenarios.mixed_quality_progression),
        ("Format Then Correctness", AdvancedScenarios.format_first_then_correctness),
        ("Difficulty Mismatch", AdvancedScenarios.difficulty_mismatch),
    ]

    for name, generator in scenarios:
        print(f"\n\nRunning: {name}")
        print("=" * 70)
        run_advanced_scenario(name, generator, num_steps=80)


# ============================================================================
# EXAMPLE 8: Custom Pattern with Randomness
# ============================================================================


def example_custom_pattern():
    """Create a custom pattern with probabilities."""
    from .training_simulator import TrainingSimulator
    import random

    random.seed(42)
    simulator = TrainingSimulator(num_generations=4)

    # 80% correct, 90% proper format, improving over time
    configs = []
    for step in range(100):
        # Improvement over time
        correct_prob = 0.5 + (step / 200)  # From 50% to 100%
        format_prob = 0.7 + (step / 500)  # From 70% to 90%

        has_think = random.random() < format_prob
        has_answer = random.random() < format_prob
        is_correct = random.random() < correct_prob

        configs.append((has_think, has_answer, is_correct))

    simulator.run_scenario(
        "Custom Pattern (Improving Over Time)",
        response_configs=configs,
    )
    simulator.print_results()


# ============================================================================
# EXAMPLE 9: Analyze Multiple Scenarios
# ============================================================================


def example_compare_scenarios():
    """Compare multiple scenarios side by side."""
    from .training_simulator import TrainingSimulator
    from .advanced_scenarios import ResponsePattern

    results = {}

    scenarios = {
        "All Perfect": [ResponsePattern.perfect()] * 60,
        "70% Correct": [
            ResponsePattern.perfect() if i % 3 == 0 else ResponsePattern.format_only()
            for i in range(60)
        ],
        "No Think Tag": [ResponsePattern.no_think()] * 60,
        "Random": [
            ResponsePattern.perfect() if i % 2 == 0 else ResponsePattern.all_bad()
            for i in range(60)
        ],
    }

    for name, configs in scenarios.items():
        sim = TrainingSimulator(num_generations=4)
        result = sim.run_scenario(name, response_configs=configs, batch_size=4)
        results[name] = {
            "final_level": result["final_level"],
            "final_success_rate": result["final_success_rate"],
            "final_step": result["final_step"],
        }

    print("\n" + "=" * 70)
    print("SCENARIO COMPARISON")
    print("=" * 70)
    for name, stats in results.items():
        print(
            f"{name:20} Level={stats['final_level']} "
            f"Success={stats['final_success_rate']:.1%} "
            f"Steps={stats['final_step']}"
        )


# ============================================================================
# EXAMPLE 10: Save and Load Results
# ============================================================================


def example_save_results():
    """Save results for later analysis."""
    from .training_simulator import TrainingSimulator
    import json

    simulator = TrainingSimulator(num_generations=4)
    simulator.run_scenario(
        "Test Scenario",
        response_configs=[(True, True, True)] * 50,
    )

    # Save results
    simulator.save_results("my_experiment.json")

    # Load and analyze
    with open("my_experiment.json") as f:
        data = json.load(f)

    print("\nLoaded results:")
    print(f"  Final Level: {data['final_level']}")
    print(f"  Final Step: {data['final_step']}")
    print(f"  Success Rate: {data['final_success_rate']:.1%}")
    print(f"  Total Snapshots: {len(data['snapshots'])}")


# ============================================================================
# EXAMPLE 12: LLM Judge as Auxiliary Reward Function
# ============================================================================


def example_llm_judge():
    """Test LLM Judge as an auxiliary reward function for math/puzzle tasks.
    
    NOTE: Requires sglang server running the Skywork Reward Model.
    Start server with:
        python -m sglang.launch_server --model-path Skywork/Reward-Preference-Alpaca-7B-v2 --port 8000
    """
    from .training_simulator import TrainingSimulator
    from .advanced_scenarios import ResponsePattern

    print("\n" + "=" * 70)
    print("LLM JUDGE AUXILIARY REWARD TEST")
    print("=" * 70)
    print("\nConfiguring simulator with LLM Judge as auxiliary reward...")
    print("This test assumes sglang server is running on localhost:8000")
    print("with Skywork Reward Model.\n")

    try:
        # Create simulator with LLM Judge enabled
        simulator = TrainingSimulator(
            num_generations=4,
            use_format=True,
            use_llm_judge=True,
            llm_judge_weight=0.2,
            llm_judge_kwargs={
                "api_host": "localhost",
                "api_port": 8000,
                "model_name": "Skywork/Reward-Preference-Alpaca-7B-v2",
            },
        )

        print("Reward Functions Configuration:")
        print(f"  ✓ use_format: {simulator.use_format}")
        print(f"  ✓ use_llm_judge: {simulator.use_llm_judge}")
        print(f"  ✓ llm_judge_weight: {simulator.llm_judge_weight}")
        print(f"  ✓ aux_weight: {simulator.aux_weight}")
        print(f"  ✓ num_generations: {simulator.num_generations}")

        # Run scenario with mixed quality responses
        print("\nRunning scenario: Mixed Quality Responses")
        print("-" * 70)

        configs = [ResponsePattern.perfect()] * 30 + [ResponsePattern.format_only()] * 20

        result = simulator.run_scenario(
            "LLM Judge Test",
            response_configs=configs,
            batch_size=4,
        )

        print("\n" + "=" * 70)
        print("RESULTS SUMMARY")
        print("=" * 70)
        print(f"Final Level: {result['final_level']}")
        print(f"Final Success Rate: {result['final_success_rate']:.1%}")
        print(f"Total Steps: {result['final_step']}")

        # Show LLM Judge scores from snapshots
        if simulator.snapshots:
            avg_judge_score = sum(s.llm_judge_score for s in simulator.snapshots) / len(
                simulator.snapshots
            )
            print(f"Average LLM Judge Score: {avg_judge_score:.3f}")

        # Save results for analysis
        simulator.save_results("llm_judge_test_results.json")
        print("\nResults saved to: llm_judge_test_results.json")

    except Exception as e:
        print(f"\n✗ Error: {e}")
        print("\nMake sure sglang server is running:")
        print("  python -m sglang.launch_server \\")
        print("    --model-path Skywork/Reward-Preference-Alpaca-7B-v2 \\")
        print("    --port 8000")


# ============================================================================
# HELPER: Response Pattern Descriptions
# ============================================================================


def print_response_patterns():
    """Print all available response patterns."""
    from .advanced_scenarios import ResponsePattern

    patterns = {
        "perfect": ResponsePattern.perfect(),
        "format_only": ResponsePattern.format_only(),
        "no_think": ResponsePattern.no_think(),
        "no_answer": ResponsePattern.no_answer(),
        "incomplete_format": ResponsePattern.incomplete_format(),
        "all_bad": ResponsePattern.all_bad(),
        "think_only": ResponsePattern.think_only(),
        "answer_only": ResponsePattern.answer_only(),
    }

    print("\nAvailable Response Patterns:")
    print("=" * 70)
    print(f"{'Pattern Name':<25} {'(think, answer, correct)':<30} {'Description'}")
    print("-" * 70)

    descriptions = {
        "perfect": "Both tags + correct answer",
        "format_only": "Both tags but wrong answer",
        "no_think": "Missing think tag",
        "no_answer": "Missing answer tag",
        "incomplete_format": "Missing both tags",
        "all_bad": "Missing format AND wrong answer",
        "think_only": "Only think tag, correct answer",
        "answer_only": "Only answer tag, wrong answer",
    }

    for name, pattern in patterns.items():
        desc = descriptions.get(name, "")
        print(f"{name:<25} {str(pattern):<30} {desc}")


# ============================================================================
# EXAMPLE 11: Comprehensive Reward Function Test (Up-Down-Up)
# ============================================================================


def example_reward_test():
    """Test all reward functions with up-down-up scenario."""
    from .test_all_rewards import example_reward_test_all_functions

    example_reward_test_all_functions()


if __name__ == "__main__":
    import sys

    examples = {
        "1": ("Perfect Responses", example_perfect),
        "2": ("Format Issues", example_format_issues),
        "3": ("Gradual Improvement", example_gradual_improvement),
        "4": ("Format Errors", example_format_errors),
        "5": ("Correctness vs Format", example_correctness_vs_format),
        "6": ("Collapse & Recovery", example_collapse_recovery),
        "7": ("Advanced Scenarios", example_advanced_scenarios),
        "8": ("Custom Pattern", example_custom_pattern),
        "9": ("Compare Scenarios", example_compare_scenarios),
        "10": ("Save Results", example_save_results),
        "11": ("Reward Function Test (All)", example_reward_test),
        "12": ("LLM Judge Auxiliary Reward", example_llm_judge),
        "patterns": ("Response Patterns", print_response_patterns),
    }

    if len(sys.argv) > 1 and sys.argv[1] in examples:
        name, func = examples[sys.argv[1]]
        print(f"\nRunning: {name}")
        print("=" * 70)
        func()
    else:
        print("\nTraining Simulator - Quick Reference Examples")
        print("=" * 70)
        print("\nUsage: python examples.py <example_number>")
        print("\nAvailable examples:")
        for num, (name, _) in examples.items():
            print(f"  {num:2} - {name}")

        print("\nExample: python examples.py 1")
        print("Example: python examples.py 11")
        print("Example: python examples.py patterns")

        # Show patterns by default
        print_response_patterns()
