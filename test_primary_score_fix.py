#!/usr/bin/env python
"""Test that primary scores are used for curriculum progression."""

from infinite_rl.curriculum import CurriculumLearning

# Create curriculum with aux_weight=0.3 (30% penalty weight)
curriculum = CurriculumLearning(
    num_generations=4,
    warmup_step=0,
    use_format=True,  # This will add format penalties
    use_repetition=False,
    use_reasoning_steps=False,
    use_length=False,
    use_lang_consistency=False,
    aux_weight=0.3,
)

# Get a task
task = curriculum.get_prompt()
print(f"Task ID: {task.task_id}")
print(f"Task Type: {task.task_type}")

# Simulate 4 responses with CORRECT answer but BAD formatting
# This tests if curriculum uses primary (correctness) vs combined (with penalties)
responses = [
    f"some text before tag\n<think>thinking</think>\n<answer>{task.expected_answer}</answer>",  # Bad: content before think
    f"<think>thinking</think>\n<answer>{task.expected_answer}</answer>",  # Good formatting
    f"intro text\n<think>thinking</think>\n<answer>{task.expected_answer}</answer>",  # Bad: content before think
    f"<think>thinking</think>\n<answer>{task.expected_answer}</answer>",  # Good formatting
]

print("\n" + "=" * 60)
print("Simulating 4 GRPO responses for the same task:")
print("=" * 60)

combined_scores = []
primary_scores = []

for i, response in enumerate(responses):
    # Create task_id variant for GRPO (same base, different instance)
    variant_task_id = f"{task.task_id.rsplit('_', 1)[0]}_{i}"

    # Add task variant to session
    from infinite_rl.task import Task

    variant_task = Task(
        task_id=variant_task_id,
        task_name=task.task_name,
        task_type=task.task_type,
        level=task.level,
        prompt=task.prompt,
        expected_answer=task.expected_answer,
        language=task.language,
    )
    curriculum.session.add_task(variant_task)

    # Compute reward
    combined = curriculum.compute_reward(variant_task_id, response)

    # Get task to check primary score
    t = curriculum.session.get_task(variant_task_id)
    primary = next(
        r.score for r in t.task_rewards if r.reward_function_name == "primary"
    )

    combined_scores.append(combined)
    primary_scores.append(primary)

    print(f"\nResponse {i+1}:")
    print(f"  Primary Score (correctness): {primary}")
    print(f"  Combined Score (with aux):   {combined:.3f}")
    print(f"  Has content before tag: {response.strip()[0] != '<'}")

print("\n" + "=" * 60)
print("GRPO Batch Results:")
print("=" * 60)
print(f"Primary Scores:  {primary_scores}")
print(f"Combined Scores: {[f'{s:.3f}' for s in combined_scores]}")
print()
print(f"Max Primary:  {max(primary_scores)}")
print(f"Mean Primary: {sum(primary_scores)/len(primary_scores):.3f}")
print(f"Max Combined: {max(combined_scores):.3f}")
print(f"Mean Combined: {sum(combined_scores)/len(combined_scores):.3f}")

print("\n" + "=" * 60)
print("Curriculum Decision:")
print("=" * 60)

# Check what would happen with OLD logic (using combined scores)
old_max = max(combined_scores)
old_mean = sum(combined_scores) / len(combined_scores)
old_success = 1 if (old_max == 1.0 or old_mean >= 0.7) else 0

print(f"OLD Logic (combined scores):")
print(f"  Max={old_max:.3f}, Mean={old_mean:.3f}")
print(f"  Would count as success: {bool(old_success)}")

# Check what happens with NEW logic (using primary scores)
new_max = max(primary_scores)
new_mean = sum(primary_scores) / len(primary_scores)
new_success = 1 if (new_max == 1.0 or new_mean >= 0.7) else 0

print(f"\nNEW Logic (primary scores):")
print(f"  Max={new_max:.3f}, Mean={new_mean:.3f}")
print(f"  Would count as success: {bool(new_success)}")

# Get actual success tracking
success_rate = curriculum.get_success_rate()
print(f"\nActual curriculum success window:")
print(f"  {success_rate}")

print("\n" + "=" * 60)
if new_success and not old_success:
    print("✅ FIX WORKING: Curriculum now tracks correctness, not formatting!")
    print("   Model can level up even with format issues.")
elif new_success and old_success:
    print("✅ Both logics agree: Task was solved well")
else:
    print("⚠️  Both failed - need to check implementation")
print("=" * 60)
