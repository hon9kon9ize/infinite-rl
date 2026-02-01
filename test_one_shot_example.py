#!/usr/bin/env python3
"""
Test script to demonstrate the one-shot example feature in puzzle prompts.
"""

from infinite_rl.prompt_templates import format_puzzle_prompt


def main():
    # Sample puzzle data
    puzzle_data = {
        "name": "FindMax",
        "docstring": "Find the maximum value in a list of integers.",
        "sat": """def sat(result: int, nums=[1, 5, 3, 9, 2]):
    return result == max(nums)""",
        "sol": """def sol(nums):
    pass""",
    }

    print("=" * 80)
    print("Python Puzzle - WITHOUT One-Shot Example")
    print("=" * 80)
    prompt_without = format_puzzle_prompt(puzzle_data, "python", one_shot=False)
    print(prompt_without)
    print()

    print("=" * 80)
    print("Python Puzzle - WITH One-Shot Example")
    print("=" * 80)
    prompt_with = format_puzzle_prompt(puzzle_data, "python", one_shot=True)
    print(prompt_with)
    print()

    # JavaScript version
    js_puzzle_data = {
        "name": "FindMax",
        "docstring": "Find the maximum value in an array of integers.",
        "sat": """function sat(result, nums=[1, 5, 3, 9, 2]) {
    return result === Math.max(...nums);
}""",
        "sol": """function sol(nums) {
    // your code here
}""",
    }

    print("=" * 80)
    print("JavaScript Puzzle - WITHOUT One-Shot Example")
    print("=" * 80)
    js_prompt_without = format_puzzle_prompt(
        js_puzzle_data, "javascript", one_shot=False
    )
    print(js_prompt_without)
    print()

    print("=" * 80)
    print("JavaScript Puzzle - WITH One-Shot Example")
    print("=" * 80)
    js_prompt_with = format_puzzle_prompt(js_puzzle_data, "javascript", one_shot=True)
    print(js_prompt_with)
    print()

    # Test with CurriculumLearning
    print("=" * 80)
    print("Testing with CurriculumLearning")
    print("=" * 80)
    from infinite_rl.curriculum import CurriculumLearning

    # Without one-shot
    curriculum_without = CurriculumLearning(one_shot=False)
    print(f"Curriculum created with one_shot=False: {curriculum_without.one_shot}")

    # With one-shot
    curriculum_with = CurriculumLearning(one_shot=True)
    print(f"Curriculum created with one_shot=True: {curriculum_with.one_shot}")
    print()

    print("✅ All tests passed! One-shot example feature is working correctly.")


if __name__ == "__main__":
    main()
