#!/usr/bin/env python3
"""
Script to generate and print a puzzle prompt with one-shot example.
Uses a sample puzzle from puzzles.json for demonstration.
"""

import json
import sys
import os
from pathlib import Path

# Add the project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from infinite_rl.prompt_templates import format_puzzle_prompt


def main():
    """Generate and print a puzzle prompt with one-shot example."""

    # Load puzzles.json
    puzzles_path = (
        project_root / "rl-data-geneator" / "infinite_rl" / "runtimes" / "puzzles.json"
    )
    with open(puzzles_path, "r") as f:
        puzzles = json.load(f)

    # Pick a sample puzzle (AbsoluteValues from javascript)
    puzzle_data = puzzles["javascript"]["AbsoluteValues"]

    print("Sample Puzzle Data:")
    print("=" * 50)
    print(f"Name: {puzzle_data['name']}")
    print(f"Language: {puzzle_data['language']}")
    print(f"Rating: {puzzle_data['rating']}")
    print(f"Description: {puzzle_data['docstring']}")
    print()

    # Generate prompt with one-shot example
    prompt = format_puzzle_prompt(puzzle_data, "javascript", one_shot=True)

    print("Generated Prompt with One-Shot Example:")
    print("=" * 50)
    print(prompt)
    print("=" * 50)

    # Print statistics
    word_count = len(prompt.split())
    char_count = len(prompt)
    print("\nPrompt Statistics:")
    print(f"Characters: {char_count:,}")
    print(f"Words: {word_count:,}")
    print(".1f")


if __name__ == "__main__":
    main()
