import argparse
import os
from infinite_rl.generator import generate_dataset


def main():
    parser = argparse.ArgumentParser(
        description="Generate synthetic RL dataset using Gemini API."
    )
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Gemini model name (e.g., gemini-1.5-flash)",
    )
    parser.add_argument(
        "--type",
        type=str,
        required=True,
        help="Comma separated types (e.g., coding,math,summarization,creativity)",
    )
    parser.add_argument(
        "--num_samples",
        type=int,
        required=True,
        help="Total number of samples to generate",
    )
    parser.add_argument(
        "--out",
        type=str,
        required=True,
        help="Output folder for dataset.csv and reward functions",
    )

    args = parser.parse_args()

    types = [t.strip() for t in args.type.split(",")]

    if not os.path.exists(args.out):
        os.makedirs(args.out)

    generate_dataset(args.model, types, args.num_samples, args.out)


if __name__ == "__main__":
    main()
