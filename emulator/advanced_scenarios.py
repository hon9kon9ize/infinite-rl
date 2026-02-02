"""
Advanced training scenario builder for curriculum learning simulation.

Provides detailed scenario analysis, custom response patterns, and
visual progression tracking.

Example scenarios:
- "Perfect": All responses have proper format and correct answers
- "Format Only": Proper format but incorrect answers
- "Degradation": Start good, gradually degrade
- "Curriculum Gap": Responses don't match current difficulty level
"""

from typing import List, Tuple, Dict, Any, Callable
from .training_simulator import TrainingSimulator


# Response configuration shortcuts
class ResponsePattern:
    """Predefined response patterns for common scenarios."""

    @staticmethod
    def perfect() -> Tuple[bool, bool, bool]:
        """Perfect response: think + answer + correct."""
        return (True, True, True)

    @staticmethod
    def format_only() -> Tuple[bool, bool, bool]:
        """Proper format but incorrect answer."""
        return (True, True, False)

    @staticmethod
    def no_think() -> Tuple[bool, bool, bool]:
        """Missing think tag."""
        return (False, True, True)

    @staticmethod
    def no_answer() -> Tuple[bool, bool, bool]:
        """Missing answer tag."""
        return (True, False, True)

    @staticmethod
    def incomplete_format() -> Tuple[bool, bool, bool]:
        """Missing both tags."""
        return (False, False, True)

    @staticmethod
    def all_bad() -> Tuple[bool, bool, bool]:
        """Missing format AND wrong answer."""
        return (False, False, False)

    @staticmethod
    def think_only() -> Tuple[bool, bool, bool]:
        """Only think tag present."""
        return (True, False, False)

    @staticmethod
    def answer_only() -> Tuple[bool, bool, bool]:
        """Only answer tag present."""
        return (False, True, False)


class AdvancedScenarios:
    """Collection of advanced training scenarios."""

    @staticmethod
    def mixed_quality_progression(
        num_steps: int = 100,
    ) -> List[Tuple[bool, bool, bool]]:
        """Simulate a realistic learning progression with mixed quality.

        Represents a model learning to format correctly and improve answer accuracy.
        """
        pattern = []

        # Phase 1: Struggling with format (0-25 steps)
        # Mix of missing tags and wrong answers
        for _ in range(25):
            if _ % 4 == 0:
                pattern.append(ResponsePattern.no_think())
            elif _ % 4 == 1:
                pattern.append(ResponsePattern.format_only())
            elif _ % 4 == 2:
                pattern.append(ResponsePattern.all_bad())
            else:
                pattern.append(ResponsePattern.perfect())

        # Phase 2: Improving format (25-50 steps)
        # Better format compliance, improving answers
        for _ in range(25):
            if _ % 3 == 0:
                pattern.append(ResponsePattern.format_only())
            else:
                pattern.append(ResponsePattern.perfect())

        # Phase 3: Good format, improving answers (50-75 steps)
        # Mostly correct but occasional mistakes
        for _ in range(25):
            if _ % 5 == 0:
                pattern.append(ResponsePattern.format_only())
            else:
                pattern.append(ResponsePattern.perfect())

        # Phase 4: Mastery (75-100 steps)
        # Consistent correctness
        for _ in range(25):
            pattern.append(ResponsePattern.perfect())

        return pattern[:num_steps]

    @staticmethod
    def format_first_then_correctness(
        num_steps: int = 80,
    ) -> List[Tuple[bool, bool, bool]]:
        """Learn formatting first, then correctness.

        Demonstrates scenario where model learns XML structure before solving tasks.
        """
        pattern = []

        # Phase 1: Learn format (0-40 steps)
        for _ in range(40):
            if _ < 20:
                # First half: mostly missing format
                pattern.append(
                    ResponsePattern.incomplete_format()
                    if _ % 3 == 0
                    else ResponsePattern.no_think()
                )
            else:
                # Second half: improving format
                pattern.append(
                    ResponsePattern.format_only()
                    if _ % 2 == 0
                    else ResponsePattern.perfect()
                )

        # Phase 2: Now learn correctness (40-80 steps)
        for _ in range(40):
            pattern.append(ResponsePattern.perfect())

        return pattern[:num_steps]

    @staticmethod
    def difficulty_mismatch(
        num_steps: int = 60,
    ) -> List[Tuple[bool, bool, bool]]:
        """Curriculum mismatch: task difficulty too hard.

        Model struggles even with proper format because task is too hard.
        """
        pattern = []

        # Phase 1: Struggling (0-30 steps)
        # Good format but wrong answers (task too hard)
        for _ in range(30):
            if _ % 5 == 0:
                pattern.append(ResponsePattern.format_only())
                pattern.append(ResponsePattern.no_think())
            else:
                pattern.append(ResponsePattern.format_only())

        # Phase 2: Sudden improvement (30-60 steps)
        # Finally grasping the concept
        for _ in range(30):
            if _ % 3 == 0:
                pattern.append(ResponsePattern.format_only())
            else:
                pattern.append(ResponsePattern.perfect())

        return pattern[:num_steps]

    @staticmethod
    def recovery_from_collapse(
        num_steps: int = 100,
    ) -> List[Tuple[bool, bool, bool]]:
        """Simulate collapse and recovery.

        Model was doing well, then fails (e.g., due to distribution shift),
        then recovers.
        """
        pattern = []

        # Phase 1: Good performance (0-30 steps)
        for _ in range(30):
            pattern.append(ResponsePattern.perfect())

        # Phase 2: Collapse (30-60 steps)
        for _ in range(30):
            if _ % 3 == 0:
                pattern.append(ResponsePattern.all_bad())
            elif _ % 3 == 1:
                pattern.append(ResponsePattern.no_think())
            else:
                pattern.append(ResponsePattern.format_only())

        # Phase 3: Recovery (60-100 steps)
        for _ in range(40):
            if _ < 20:
                # Slow recovery
                pattern.append(
                    ResponsePattern.format_only()
                    if _ % 2 == 0
                    else ResponsePattern.perfect()
                )
            else:
                # Full recovery
                pattern.append(ResponsePattern.perfect())

        return pattern[:num_steps]

    @staticmethod
    def cascade_success(
        num_steps: int = 100,
    ) -> List[Tuple[bool, bool, bool]]:
        """Demonstrate cascading effect: success breeds more success.

        As model improves at lower levels, it naturally progresses.
        """
        pattern = []

        # Level 0: Quick mastery
        for _ in range(15):
            pattern.append(
                ResponsePattern.perfect() if _ > 5 else ResponsePattern.format_only()
            )

        # Level 1: Gets harder, takes longer
        for _ in range(25):
            if _ < 10:
                pattern.append(
                    ResponsePattern.format_only()
                    if _ % 2 == 0
                    else ResponsePattern.perfect()
                )
            else:
                pattern.append(ResponsePattern.perfect())

        # Level 2: Even harder
        for _ in range(30):
            pattern.append(
                ResponsePattern.perfect() if _ > 15 else ResponsePattern.format_only()
            )

        # Continued success at higher levels
        for _ in range(30):
            pattern.append(ResponsePattern.perfect())

        return pattern[:num_steps]


def run_advanced_scenario(
    name: str,
    pattern_generator: Callable[[], List[Tuple[bool, bool, bool]]],
    num_steps: int = 100,
    num_generations: int = 4,
):
    """Run an advanced scenario and display results.

    Args:
        name: Scenario name
        pattern_generator: Function that generates response patterns
        num_steps: Total number of steps to simulate
        num_generations: Responses per GRPO batch
    """
    simulator = TrainingSimulator(
        num_generations=num_generations,
        use_format=True,
        success_rate_threshold=0.7,
    )

    pattern = pattern_generator()
    if len(pattern) < num_steps:
        pattern = pattern * ((num_steps // len(pattern)) + 1)

    simulator.run_scenario(
        name,
        response_configs=pattern[:num_steps],
        batch_size=num_generations,
    )

    simulator.print_results()
    simulator.save_results(f"results_{name.lower().replace(' ', '_')}.json")


if __name__ == "__main__":
    print("\n" + "=" * 70)
    print("ADVANCED CURRICULUM LEARNING SCENARIOS")
    print("=" * 70)

    scenarios = [
        ("Mixed Quality Progression", AdvancedScenarios.mixed_quality_progression),
        (
            "Format First, Then Correctness",
            AdvancedScenarios.format_first_then_correctness,
        ),
        ("Difficulty Mismatch", AdvancedScenarios.difficulty_mismatch),
        ("Recovery from Collapse", AdvancedScenarios.recovery_from_collapse),
        ("Cascade Success", AdvancedScenarios.cascade_success),
    ]

    for scenario_name, pattern_func in scenarios:
        try:
            run_advanced_scenario(scenario_name, pattern_func, num_steps=100)
        except Exception as e:
            print(f"\n❌ Error running scenario '{scenario_name}': {e}")
            import traceback

            traceback.print_exc()
