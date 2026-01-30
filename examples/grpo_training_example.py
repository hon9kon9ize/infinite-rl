#!/usr/bin/env python3
"""
Example: GRPO training with updated CurriculumLearning.

Shows how the curriculum automatically handles batch-level success tracking
when you call compute_reward() 4 times with different responses for the same prompt.
"""

from infinite_rl.curriculum import CurriculumLearning
import json


def example_grpo_training_loop():
    """Simulate a GRPO training loop with 4 responses per prompt."""

    # Initialize curriculum with GRPO-optimized settings
    curriculum = CurriculumLearning(
        log_file="grpo_training.jsonl",
        window_size=10,
        success_rate_threshold=0.8,  # Advance at 80% success rate
        demote_threshold=0.3,  # Demote at 30% success rate
        warmup_step=20,  # Run 20 prompts at level 0 before puzzles
        reflective_learning_rate=0.2,  # 20% chance of reflective learning
    )

    print("=" * 70)
    print("GRPO Training with Curriculum Learning")
    print("=" * 70)
    print(f"Starting at Level: {curriculum.current_level}")
    print(f"Warmup steps: {curriculum.warmup_step}")
    print(f"Success threshold: {curriculum.success_rate_threshold:.0%}")
    print()

    # Simulate 5 training iterations (prompts)
    for prompt_idx in range(5):
        # Get a new task prompt
        task = curriculum.get_prompt()
        if task is None:
            print("❌ No tasks available")
            break

        print(f"\n{'='*70}")
        print(f"Prompt {prompt_idx + 1}: {task.task_id}")
        print(f"Task type: {task.task_type}, Level: {task.level}")
        print(f"Current curriculum level: {curriculum.current_level}")
        print(f"Is warmup: {curriculum.is_warmup()}")

        # Simulate 4 GRPO responses for this prompt
        responses_scores = []
        print(f"\nGenerating 4 responses for this prompt...")

        for response_idx in range(4):
            # Simulate response quality (this is where your model output goes)
            # For demo: vary quality to show how batching works
            if response_idx == 0:
                simulated_score = 1.0  # Good response
            elif response_idx == 1:
                simulated_score = 0.6  # Okay response
            elif response_idx == 2:
                simulated_score = 0.4  # Poor response
            else:
                simulated_score = 0.2  # Very poor response

            responses_scores.append(simulated_score)

            # Compute reward (this is called once per response in GRPO)
            # In reality, you'd get this from your model
            task_id_with_idx = f"{task.task_id}_resp{response_idx}"
            reward = curriculum.compute_reward(
                task_id_with_idx, f"Response {response_idx}"
            )

            print(
                f"  Response {response_idx + 1}: score={simulated_score:.1f}, "
                f"combined_reward={reward:.3f}"
            )

        # Check batch statistics
        mean_score = sum(responses_scores) / len(responses_scores)
        max_score = max(responses_scores)
        print(f"\nBatch statistics:")
        print(f"  Mean score: {mean_score:.2f}")
        print(f"  Max score: {max_score:.2f}")
        print(
            f"  Prompt success: {1 if (max_score == 1.0 or mean_score >= 0.7) else 0}"
        )

        # Show current learning state
        stats = curriculum.get_learning_stats()
        print(f"\nCurriculum state:")
        print(f"  Current level: {stats['current_level']}")
        print(
            f"  Success rate: {stats['sliding_window_stats'].get('mean_success_rate', 0):.0%}"
        )

    # Final summary
    print(f"\n{'='*70}")
    print("Training Summary")
    print(f"{'='*70}")
    final_stats = curriculum.get_learning_stats()
    print(f"Final level: {final_stats['current_level']}")
    print(f"Total prompts trained: {curriculum.global_step}")
    print(f"Task type success rates:")
    for task_stat in final_stats["sliding_window_stats"].get("by_task_type", []):
        print(
            f"  {task_stat['task_type']}: {task_stat['success_rate']:.0%} "
            f"({task_stat['samples']} prompts)"
        )


if __name__ == "__main__":
    example_grpo_training_loop()
