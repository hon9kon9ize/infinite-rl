#!/usr/bin/env python3

import argparse
import json
import os
import sys
from typing import List

# Add the project root to the path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from infinite_rl.puzzle_generator import PuzzleDatasetGenerator


def generate_dataset(
    num_samples: int, out_dir: str, task_dist: List[float], debug: bool = False
):
    """Generate a dataset of RLHF samples."""
    generator = PuzzleDatasetGenerator()
    samples = generator.generate_dataset(num_samples, task_dist)

    # For now, just print the samples
    for sample in samples:
        print(json.dumps(sample))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate synthetic RLHF dataset")
    parser.add_argument(
        "--num_samples", type=int, default=100, help="Number of samples to generate"
    )
    parser.add_argument(
        "--out_dir", type=str, default="./data", help="Output directory"
    )
    parser.add_argument(
        "--task_dist",
        type=str,
        default="1.0",
        help="Task distribution (comma-separated floats)",
    )
    parser.add_argument("--debug", action="store_true", help="Enable debug mode")

    args = parser.parse_args()

    task_dist = [float(x) for x in args.task_dist.split(",")]

    generate_dataset(args.num_samples, args.out_dir, task_dist, args.debug)
