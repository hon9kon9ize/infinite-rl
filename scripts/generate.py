import argparse
import os
import sys

# Ensure the root directory is in the python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from infinite_rl.generator import generate_dataset


def main():
    parser = argparse.ArgumentParser(
        description="Generate RL synthetic dataset using Gemini"
    )
    parser.add_argument(
        "--num_samples", type=int, default=10, help="Number of samples to generate"
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="gemini-2.0-flash-exp",
        help="Gemini model name",
    )
    parser.add_argument(
        "--out_dir", type=str, default="data", help="Output directory for the dataset"
    )
    parser.add_argument(
        "--save_every",
        type=int,
        default=1,
        help="How often to save progress to CSV (default: 1)",
    )
    parser.add_argument(
        "--max_retries",
        type=int,
        default=5,
        help="Maximum consecutive retries before stopping (default: 5)",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=5,
        help="Timeout (in seconds) for reward function execution (default: 5)",
    )
    parser.add_argument(
        "--task_dist",
        type=str,
        default="0.5,0.1,0.3,0.1",
        help="Task distribution [code, html, math, summarization] (default: 0.5,0.1,0.3,0.1)",
    )

    args = parser.parse_args()

    # Create output directory if it doesn't exist
    if not os.path.exists(args.out_dir):
        os.makedirs(args.out_dir)
        print(f"Created output directory: {args.out_dir}")

    try:
        generate_dataset(
            model_name=args.model_name,
            num_samples=args.num_samples,
            out_dir=args.out_dir,
            save_every=args.save_every,
            max_retries=args.max_retries,
            timeout=args.timeout,
            task_dist=args.task_dist,
        )
    except Exception as e:
        print(f"Error during generation: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
