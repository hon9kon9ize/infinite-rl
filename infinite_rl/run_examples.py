import os
import json
from pathlib import Path
from .parser import ExampleParser
from .generator import get_reward_function


def run_examples():
    """
    Loads and runs all examples from the examples/ directory.
    Primarily used for testing library installation and verifying reward functions.
    """
    # Find examples directory relative to the package installation
    package_dir = os.path.dirname(os.path.abspath(__file__))
    examples_dir = os.path.join(package_dir, "examples")

    if not os.path.exists(examples_dir):
        # Fallback check for current working directory if not found in package
        cwd_examples = os.path.join(os.getcwd(), "examples")
        if os.path.exists(cwd_examples):
            examples_dir = cwd_examples
        else:
            print(f"Error: Examples directory not found at {examples_dir}")
            return

    print(f"Loading examples from: {examples_dir}")
    examples = ExampleParser.get_all_examples(examples_dir)

    results = []
    print("\n" + "=" * 50)
    print(f"{'Example Name':<20} | {'Status':<10} | {'Score':<10}")
    print("-" * 50)

    for name, data in examples.items():
        # Map filename to task type
        task_type = name.lower()
        if task_type in ["js"]:
            task_type = "javascript"
        if task_type in ["ts"]:
            task_type = "typescript"

        reward_fn = get_reward_function(task_type)
        if not reward_fn:
            # Fallback if specific language name doesn't match directly
            reward_fn = get_reward_function("coding")
            if "python" in name.lower():
                reward_fn.set_language("python")
            elif "javascript" in name.lower():
                reward_fn.set_language("javascript")

        if not reward_fn:
            print(f"{name:<20} | Skip       | N/A")
            continue

        try:
            reward_fn.initialize()

            # Preparation for specific reward functions
            kwargs = {}

            # HTML and some coding tasks might need expected output to be parsed if it's JSON
            expected_output = data["answer"]
            if task_type == "html":
                try:
                    expected_output = json.loads(data["answer"])
                except:
                    pass

            score = reward_fn.compute_reward(
                data["response"], expected_output, **kwargs
            )
            total_score = (score.format_score + score.correctness_score) / 2.0

            status = "PASS" if total_score > 0.8 else "FAIL"
            print(f"{name:<20} | {status:<10} | {total_score:<10.2f}")

            if total_score < 0.8 and score.error_msg:
                print(f"  └─ Error: {score.error_msg}")

            results.append(total_score)
        except Exception as e:
            print(f"{name:<20} | ERROR      | 0.00")
            print(f"  └─ Exception: {e}")
            results.append(0.0)

    print("-" * 50)
    if results:
        avg_score = sum(results) / len(results)
        print(f"Average Score: {avg_score:.2f}")
    print("=" * 50 + "\n")


if __name__ == "__main__":
    run_examples()
