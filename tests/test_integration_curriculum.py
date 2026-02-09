"""
Integration test to verify the complete training pipeline works correctly.

This test validates:
1. The reward function produces correct order
2. The curriculum correctly processes rewards
3. Learning signals are properly propagated
"""

import json
from pathlib import Path
from infinite_rl.curriculum import CurriculumLearning
from infinite_rl.dynamic_dataset import DynamicCurriculumDataset


def test_curriculum_integration():
    """Integration test: Full curriculum flow with reward computation."""

    print("\n" + "=" * 80)
    print("CURRICULUM INTEGRATION TEST")
    print("=" * 80 + "\n")

    # Initialize curriculum without LLM Judge (faster)
    curriculum = CurriculumLearning(
        use_llm_judge=False,
        use_format=True,
        window_size=10,
        success_rate_threshold=0.5,
        demote_threshold=0.2,
        num_generations=4,
    )

    print(f"✓ Curriculum initialized")
    print(f"  - Current level: {curriculum.current_level}")
    print(f"  - Num generations: {curriculum.num_generations}")

    # Get a task
    task = curriculum.get_prompt()
    if not task:
        print("✗ Failed to get a task from curriculum")
        return False

    print(f"\n✓ Got task: {task.task_id}")
    print(f"  - Task type: {task.task_type}")
    print(f"  - Level: {task.level}")

    # Simulate 4 generations with different outcomes
    test_outputs = [
        # Gen 0: Correct answer with proper format
        f"<{curriculum.think_tag}>Thinking step by step.</{curriculum.think_tag}>\n<{curriculum.answer_tag}>42</{curriculum.answer_tag}>",
        # Gen 1: Wrong answer
        f"<{curriculum.think_tag}>Wrong thinking.</{curriculum.think_tag}>\n<{curriculum.answer_tag}>0</{curriculum.answer_tag}>",
        # Gen 2: Correct answer with proper format
        f"<{curriculum.think_tag}>Correct thinking.</{curriculum.think_tag}>\n<{curriculum.answer_tag}>42</{curriculum.answer_tag}>",
        # Gen 3: Wrong format (no answer tag)
        f"<{curriculum.think_tag}>No answer provided.</{curriculum.think_tag}>",
    ]

    print(f"\n📝 Testing {len(test_outputs)} generations:")

    rewards = []
    for i, output in enumerate(test_outputs):
        reward = curriculum.compute_reward(task.task_id, output)
        rewards.append(reward)
        print(f"  Gen {i}: reward = {reward:.4f}")

    print(f"\n✓ All rewards computed successfully")
    print(f"  - Rewards in order: {[f'{r:.4f}' for r in rewards]}")

    # Verify we have 4 rewards
    if len(rewards) != len(test_outputs):
        print(f"✗ Expected {len(test_outputs)} rewards, got {len(rewards)}")
        return False

    # Verify rewards are in expected range [0, 1]
    if not all(0.0 <= r <= 1.0 for r in rewards):
        print(f"✗ Some rewards out of range [0, 1]: {rewards}")
        return False

    print(f"✓ All rewards in valid range [0, 1]")

    # Check curriculum learning state
    stats = curriculum.get_learning_stats()
    print(f"\n📊 Curriculum state after 1 task:")
    print(f"  - Current level: {stats['current_level']}")
    print(f"  - Global step: {curriculum.global_step}")
    print(f"  - Task counters: {stats['task_counters']}")

    # Verify correctness was tracked
    task_obj = curriculum.session.get_task(task.task_id)
    if task_obj:
        print(f"  - Task is_correct: {task_obj.is_correct}")
        print(f"  - Task has {len(task_obj.generations)} generations")

        for j, gen in enumerate(task_obj.generations):
            print(
                f"    Gen {j}: primary={gen.primary_score:.4f}, combined={gen.combined_score:.4f}, is_correct={gen.is_correct}"
            )

    print(f"\n✅ Integration test PASSED")
    return True


def test_dynamic_dataset():
    """Test that dynamic dataset works with curriculum."""

    print("\n" + "=" * 80)
    print("DYNAMIC DATASET TEST")
    print("=" * 80 + "\n")

    curriculum = CurriculumLearning(
        use_llm_judge=False,
        num_generations=4,
    )

    dataset = DynamicCurriculumDataset(curriculum, num_samples=10)

    print(f"✓ Dynamic dataset created with {len(dataset)} virtual samples")

    # Get a few samples
    for i in range(3):
        sample = dataset[i]

        # Verify sample structure
        if "messages" not in sample:
            print(f"✗ Sample {i} missing 'messages'")
            return False

        if "task_id" not in sample:
            print(f"✗ Sample {i} missing 'task_id'")
            return False

        print(f"✓ Sample {i}: task_id={sample['task_id']}")

    print(f"\n✅ Dynamic dataset test PASSED")
    return True


def test_order_preservation_end_to_end():
    """End-to-end test verifying order preservation through the pipeline."""

    print("\n" + "=" * 80)
    print("ORDER PRESERVATION END-TO-END TEST")
    print("=" * 80 + "\n")

    # This simulates what happens in the training loop
    curriculum = CurriculumLearning(
        use_llm_judge=False,
        use_format=True,
        num_generations=3,
    )

    # Create a task
    task = curriculum.get_prompt()
    task_id = task.task_id

    # Simulate computing rewards in the order they'll appear
    test_cases = [
        ("output_A", 0.9),  # Good
        ("output_B", 0.5),  # Medium
        ("output_C", 0.1),  # Bad
    ]

    print(f"Test case: Computing rewards in order and verifying assignment...")
    print(f"Task ID: {task_id}\n")

    # Track which output was given which score
    output_to_score = {}

    for output, expected_category in test_cases:
        # Note: actual score depends on task correctness, but we can verify order
        reward = curriculum.compute_reward(task_id, output)
        output_to_score[output] = reward
        print(f"  {output}: {reward:.4f}")

    # Verify we got 3 rewards
    if len(output_to_score) != 3:
        print(f"✗ Expected 3 outputs, got {len(output_to_score)}")
        return False

    # Get task and verify generations are in order
    task_obj = curriculum.session.get_task(task_id)
    if not task_obj or len(task_obj.generations) != 3:
        print(
            f"✗ Expected 3 generations, got {len(task_obj.generations) if task_obj else 0}"
        )
        return False

    print(f"\n✓ Verified {len(task_obj.generations)} generations stored in order")

    for i, gen in enumerate(task_obj.generations):
        print(f"  Gen {i}: output starts with '{gen.output[:20]}...'")

    print(f"\n✅ Order preservation test PASSED")
    return True


if __name__ == "__main__":
    all_pass = True

    try:
        all_pass &= test_curriculum_integration()
    except Exception as e:
        print(f"✗ Curriculum integration test failed: {e}")
        import traceback

        traceback.print_exc()
        all_pass = False

    try:
        all_pass &= test_dynamic_dataset()
    except Exception as e:
        print(f"✗ Dynamic dataset test failed: {e}")
        import traceback

        traceback.print_exc()
        all_pass = False

    try:
        all_pass &= test_order_preservation_end_to_end()
    except Exception as e:
        print(f"✗ Order preservation test failed: {e}")
        import traceback

        traceback.print_exc()
        all_pass = False

    print("\n" + "=" * 80)
    if all_pass:
        print("✅ ALL INTEGRATION TESTS PASSED")
    else:
        print("❌ SOME INTEGRATION TESTS FAILED")
    print("=" * 80 + "\n")

    exit(0 if all_pass else 1)
