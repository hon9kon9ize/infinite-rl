#!/usr/bin/env python3
"""Test auxiliary reward functions in curriculum learning."""

from infinite_rl.curriculum import CurriculumLearning


def test_default_initialization():
    """Test default initialization (format=True only)."""
    print("=== Test 1: Default initialization ===")
    cl = CurriculumLearning()
    aux_fns = list(cl.aux_reward_functions.keys())
    print(f"Auxiliary functions: {aux_fns}")
    assert "format" in aux_fns, "Format should be enabled by default"
    assert (
        "lang_consistency" not in aux_fns
    ), "Lang consistency should be disabled by default"
    assert "repetition" not in aux_fns, "Repetition should be disabled by default"
    print("✓ Passed\n")


def test_enable_all_aux_rewards():
    """Test enabling all auxiliary rewards."""
    print("=== Test 2: All auxiliary rewards enabled ===")
    cl = CurriculumLearning(
        use_lang_consistency=True,
        use_repetition=True,
        use_format=True,
    )
    aux_fns = set(cl.aux_reward_functions.keys())
    print(f"Auxiliary functions: {aux_fns}")
    assert aux_fns == {
        "format",
        "lang_consistency",
        "repetition",
    }, f"Expected all three, got {aux_fns}"
    print("✓ Passed\n")


def test_disable_format():
    """Test disabling format reward."""
    print("=== Test 3: Disable format ===")
    cl = CurriculumLearning(
        use_lang_consistency=True,
        use_repetition=True,
        use_format=False,
    )
    aux_fns = set(cl.aux_reward_functions.keys())
    print(f"Auxiliary functions: {aux_fns}")
    assert aux_fns == {
        "lang_consistency",
        "repetition",
    }, f"Expected lang_consistency and repetition, got {aux_fns}"
    print("✓ Passed\n")


def test_custom_kwargs():
    """Test passing custom kwargs to auxiliary functions."""
    print("=== Test 4: Custom kwargs ===")
    cl = CurriculumLearning(
        use_repetition=True,
        use_format=False,
        repetition_kwargs={"weight": -0.2},
    )
    aux_fns = list(cl.aux_reward_functions.keys())
    print(f"Auxiliary functions: {aux_fns}")
    assert "repetition" in aux_fns, "Repetition should be enabled"
    assert "format" not in aux_fns, "Format should be disabled"
    print("✓ Passed\n")


def test_learning_stats_includes_aux():
    """Test that learning stats include auxiliary reward functions."""
    print("=== Test 5: Learning stats includes aux functions ===")
    cl = CurriculumLearning(
        use_lang_consistency=True,
        use_format=True,
    )
    stats = cl.get_learning_stats()
    print(f"Stats keys: {stats.keys()}")
    assert "aux_reward_functions" in stats, "Stats should include aux_reward_functions"
    print(f"Aux functions in stats: {stats['aux_reward_functions']}")
    print("✓ Passed\n")


def test_get_aux_reward_scores():
    """Test getting auxiliary reward scores."""
    print("=== Test 6: Get auxiliary reward scores ===")
    cl = CurriculumLearning(
        use_format=True,
        use_repetition=False,
    )

    # Get a task
    task = cl.get_prompt()
    if task:
        print(f"Task type: {task['task_type']}")

        # Create response
        if task["task_type"] == "puzzle":
            response = "<reasoning>test</reasoning><solution>```javascript\nfunction sol() { return 'test'; }\n```</solution>"
        else:
            response = f"<reasoning>test</reasoning><solution>{task.get('expected_output', '0')}</solution>"

        # Get aux scores
        aux_scores = cl.get_aux_reward_scores(response, task["expected_output"])
        print(f"Auxiliary scores: {aux_scores}")
        assert isinstance(aux_scores, dict), "Should return a dictionary"
        assert "format" in aux_scores, "Should have format score"
        print("✓ Passed\n")
    else:
        print("No tasks available, skipping test\n")


def test_combined_reward_computation():
    """Test combined reward computation with auxiliary functions."""
    print("=== Test 7: Combined reward computation ===")
    cl = CurriculumLearning(
        use_format=True,
        use_lang_consistency=False,
    )

    task = cl.get_prompt()
    if task:
        print(f"Task type: {task['task_type']}")

        # Create response
        if task["task_type"] == "puzzle":
            response = "<reasoning>test</reasoning><solution>```javascript\nfunction sol() { return 'test'; }\n```</solution>"
        else:
            response = f"<reasoning>test</reasoning><solution>{task.get('expected_output', '0')}</solution>"

        # Get combined reward
        reward = cl.compute_reward(
            task_type=task["task_type"],
            model_output=response,
            expected_output=task["expected_output"],
            task_id="test_task",
        )

        print(f"Combined reward: {reward}")
        assert isinstance(reward, float), "Reward should be a float"
        assert 0 <= reward <= 1, f"Reward should be between 0 and 1, got {reward}"
        print("✓ Passed\n")
    else:
        print("No tasks available, skipping test\n")


if __name__ == "__main__":
    print("Testing Auxiliary Reward Functions in Curriculum Learning\n")
    print("=" * 60 + "\n")

    test_default_initialization()
    test_enable_all_aux_rewards()
    test_disable_format()
    test_custom_kwargs()
    test_learning_stats_includes_aux()
    test_get_aux_reward_scores()
    test_combined_reward_computation()

    print("=" * 60)
    print("✅ All tests passed!")
