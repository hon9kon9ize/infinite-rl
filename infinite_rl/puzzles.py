"""
Puzzle prompt management for Infinite RL.

This module loads puzzle prompt assets and provides functions to retrieve
prompts for specific puzzles.
"""

import json
from pathlib import Path
from typing import Dict, Optional


class PuzzlePrompts:
    """Manager for puzzle prompts loaded from JSON assets."""

    def __init__(self):
        self._prompts: Dict[str, Dict] = {}
        self._loaded = False

    def _load_prompts(self):
        """Load all puzzle prompts from JSON file."""
        if self._loaded:
            return

        # Hardcoded path to puzzles.json
        import os
        puzzles_file = Path("/Users/josephcheng/Projects/rl-data-geneator/infinite_rl/runtimes/puzzles.json")

        if not puzzles_file.exists():
            # Find the puzzles JSON file in the runtimes directory
            package_dir = Path(__file__).parent
            puzzles_file = package_dir / "runtimes" / "puzzles.json"

        print(f"Looking for puzzles.json at: {puzzles_file}")
        print(f"File exists: {puzzles_file.exists()}")

        if not puzzles_file.exists():
            # Fallback: try in package directory
            puzzles_file = package_dir / "puzzles.json"
            print(f"Trying fallback location: {puzzles_file}")
            print(f"File exists: {puzzles_file.exists()}")

        if not puzzles_file.exists():
            # Another fallback: try relative to current file
            puzzles_file = Path(__file__).parent.parent / "puzzles.json"
            print(f"Trying second fallback: {puzzles_file}")
            print(f"File exists: {puzzles_file.exists()}")

        if not puzzles_file.exists():
            print("Could not find puzzles.json file")
            return

        try:
            print(f"Loading puzzles from: {puzzles_file}")
            with open(puzzles_file, "r", encoding="utf-8") as f:
                all_puzzles = json.load(f)

            print(f"Loaded puzzle languages: {list(all_puzzles.keys())}")
            for language, puzzles in all_puzzles.items():
                print(f"  {language}: {len(puzzles)} puzzles")
                for puzzle_name, data in puzzles.items():
                    self._prompts[f"{language}/{puzzle_name}"] = data

            print(f"Total puzzles loaded: {len(self._prompts)}")

        except Exception as e:
            print(f"Error loading puzzles: {e}")
            raise  # Re-raise the exception

        self._loaded = True

    def get_puzzle_data(
        self, puzzle_name: str, language: str = "javascript"
    ) -> Optional[Dict]:
        """
        Get the full data dict for a specific puzzle.

        Args:
            puzzle_name: Name of the puzzle (e.g., "QuadraticRoot")
            language: Language ("javascript" or "python")

        Returns:
            The full puzzle data dict, or None if not found
        """
        self._load_prompts()
        key = f"{language}/{puzzle_name}"
        return self._prompts.get(key)

    def get_available_puzzles(self, language: str = "javascript") -> list:
        """
        Get list of available puzzles for a language.

        Args:
            language: Language ("javascript" or "python")

        Returns:
            List of puzzle names
        """
        self._load_prompts()
        return [
            key.split("/", 1)[1]
            for key in self._prompts.keys()
            if key.startswith(f"{language}/")
        ]


# Global instance
_puzzle_prompts = PuzzlePrompts()


def get_puzzle_data(puzzle_name: str, language: str = "javascript") -> Optional[Dict]:
    """
    Get the full data dict for a specific puzzle.

    Args:
        puzzle_name: Name of the puzzle (e.g., "QuadraticRoot")
        language: Language ("javascript" or "python")

    Returns:
        The full puzzle data dict, or None if not found
    """
    return _puzzle_prompts.get_puzzle_data(puzzle_name, language)


def get_available_puzzles(language: str = "javascript") -> list:
    """
    Get list of available puzzles for a language.

    Args:
        language: Language ("javascript" or "python")

    Returns:
        List of puzzle names
    """
    return _puzzle_prompts.get_available_puzzles(language)
