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

    # Pre-create reward functions registry to reuse across examples
    from .reward_functions import get_reward_functions

    reward_fns = get_reward_functions(timeout=5)
    format_fn = reward_fns.get("format")

    for name, data in examples.items():
        # Map filename to task type
        task_type = name.lower()

        try:
            expected_output = data["answer"]
            model_output = data["response"]

            # Main task function (fall back to generic if task not found)
            main_fn = reward_fns.get(task_type)
            if main_fn is None:
                print(
                    f"{name:<20} | SKIP       | 0.00  (no reward fn for '{task_type}')"
                )
                continue

            main_score = main_fn.compute_reward(model_output, expected_output)
            fmt_score = (
                format_fn.compute_reward(model_output, None) if format_fn else format_fn
            )

            main_val = float(getattr(main_score, "score", 0.0))
            fmt_val = float(getattr(fmt_score, "score", 0.0)) if fmt_score else 0.0

            total_score = (fmt_val + main_val) / 2.0

            # Some rewards are aux-only (e.g., repetition/length). If main_val low but aux high, consider PASS
            aux_val = (
                float(getattr(main_score, "aux_score", 0.0))
                if hasattr(main_score, "aux_score")
                else 0.0
            )
            if total_score < 0.8 and aux_val >= 0.8:
                total_score = aux_val
                status = "PASS"
            else:
                status = "PASS" if total_score > 0.8 else "FAIL"

            print(f"{name:<20} | {status:<10} | {total_score:<10.2f}")

            if status == "FAIL" and getattr(main_score, "error_msg", None):
                em = main_score.error_msg
                if isinstance(em, dict):
                    print(f"  └─ Error: {"\n".join(em.values())}")
                else:
                    print(f"  └─ Error: {em}")

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
