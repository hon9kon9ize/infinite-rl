"""
Example: Using Reflective Learning in Infinite RL

This example demonstrates how to use the reflective learning feature to help
models improve their formatting when they fail format validation.
"""

from infinite_rl import CurriculumLearning
from infinite_rl.task import Task
from infinite_rl.reward_functions import RewardFunctionScore


def main():
    # Create a curriculum with reflective learning enabled (10% chance)
    cl = CurriculumLearning(
        reflective_learning_rate=0.1,  # 10% chance to trigger reflective learning
        warmup_step=32,  # Regular warmup stage
        timeout=10,
    )

    print("Reflective Learning Example")
    print("=" * 50)
    print(f"Reflective learning rate: {cl.reflective_learning_rate}")
    print(
        f"Reflective learning is {'ENABLED' if cl.reflective_learning_rate > 0 else 'DISABLED'}"
    )
    print()

    # Example 1: Simulate a format failure
    print("Example 1: Format Failure Detection")
    print("-" * 50)

    # Create a task that was attempted but failed format validation
    failed_task = Task(
        task_id="math_001",
        task_name="Addition Problem",
        task_type="math",
        level=1,
        prompt="What is 15 + 27?",
        expected_answer="42",
        model_output="The answer is 42",  # Missing <answer> tags!
    )

    # Add a format failure reward (score=0 means failed format)
    format_fail_reward = RewardFunctionScore(
        score=0.0,
        reward_function_name="format",
        info="Missing <answer>...</answer> tags",
    )
    failed_task.add_reward(format_fail_reward)
    cl.session.add_task(failed_task)

    print(f"Task: {failed_task.task_name}")
    print(f"Model output: '{failed_task.model_output}'")
    print(f"Format validation: FAILED ❌")
    print()

    # Example 2: Get reflective tasks
    print("Example 2: Generating Reflective Prompts")
    print("-" * 50)

    # Exit warmup to allow reflective learning
    cl.global_step = cl.warmup_step

    # Try to get a reflective prompt
    format_failures = cl._get_format_failure_tasks()
    print(f"Found {len(format_failures)} format failure(s)")

    if format_failures:
        print("\nGenerating reflective prompt for the first failure...")
        reflective_task = cl._get_reflective_prompt()

        if reflective_task:
            print(f"\n✅ Reflective task created: {reflective_task.task_name}")
            print(f"Task ID: {reflective_task.task_id}")
            print(f"\nReflective Prompt Preview:")
            print("-" * 50)
            # Show first 500 chars of the prompt
            prompt_preview = (
                reflective_task.prompt[:500] + "..."
                if len(reflective_task.prompt) > 500
                else reflective_task.prompt
            )
            print(prompt_preview)
            print("-" * 50)
    print()

    # Example 3: Reflective learning during task selection
    print("Example 3: Task Selection with Reflective Learning")
    print("-" * 50)
    print(f"Current level: {cl.current_level}")
    print(f"Global step: {cl.global_step} (warmup enabled until step {cl.warmup_step})")
    print()

    # Show how reflective learning works
    print("When get_prompt() is called:")
    print("  1. If reflective_learning_rate = 0.0: Always returns normal tasks")
    print("  2. If reflective_learning_rate > 0: May return format failure tasks:")
    print("     - Trigger probability = reflective_learning_rate")
    print("     - Only triggers after warmup phase (global_step >= warmup_step)")
    print("     - Only if format failures exist in session")
    print()
    print("Example probabilities:")
    print("  - reflective_learning_rate=0.0  → 0% chance")
    print("  - reflective_learning_rate=0.1  → 10% chance")
    print("  - reflective_learning_rate=0.3  → 30% chance")
    print("  - reflective_learning_rate=1.0  → 100% chance (always)")
    print()

    # Example 4: Configuration recommendations
    print("Example 4: Configuration Recommendations")
    print("-" * 50)
    print("Disable reflective learning:")
    print("  cl = CurriculumLearning(reflective_learning_rate=0.0)")
    print()
    print("Mild reflective learning (occasional):")
    print("  cl = CurriculumLearning(reflective_learning_rate=0.05)  # 5%")
    print()
    print("Standard reflective learning (recommended):")
    print("  cl = CurriculumLearning(reflective_learning_rate=0.1)   # 10%")
    print()
    print("Aggressive reflective learning:")
    print("  cl = CurriculumLearning(reflective_learning_rate=0.3)   # 30%")
    print()
    print("Always use reflective learning (for debugging):")
    print("  cl = CurriculumLearning(reflective_learning_rate=1.0)   # 100%")
    print()

    # Example 5: Integration with reward computation
    print("Example 5: Full Integration with Reward Computation")
    print("-" * 50)
    print("In practice, reflective learning works like this:")
    print()
    print("  1. Model attempts task → get_prompt() called")
    print("     └─> Might return reflective task (if conditions met)")
    print()
    print("  2. Model generates response → compute_reward() called")
    print("     └─> Evaluates format with FormatRewardFunction")
    print("     └─> Stores format failures in session")
    print()
    print("  3. Next task selection → get_prompt() called again")
    print("     └─> May select from format failures")
    print("     └─> Wraps in reflective prompt template")
    print("     └─> Model retries with format guidance")
    print()


if __name__ == "__main__":
    main()
