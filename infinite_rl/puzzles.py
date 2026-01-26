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

        # Find the puzzles JSON file in the runtimes directory
        package_dir = Path(__file__).parent
        puzzles_file = package_dir / "runtimes" / "puzzles.json"

        if not puzzles_file.exists():
            # Fallback: try in package directory
            puzzles_file = package_dir / "puzzles.json"

        if not puzzles_file.exists():
            # Another fallback: try relative to current file
            puzzles_file = Path(__file__).parent.parent / "puzzles.json"

        if not puzzles_file.exists():
            return

        try:
            with open(puzzles_file, "r", encoding="utf-8") as f:
                all_puzzles = json.load(f)

            for language, puzzles in all_puzzles.items():
                for puzzle_name, data in puzzles.items():
                    self._prompts[f"{language}/{puzzle_name}"] = data

        except Exception:
            pass

        self._loaded = True

    def get_puzzle_prompt(
        self, puzzle_name: str, language: str = "javascript"
    ) -> Optional[str]:
        """
        Get the prompt for a specific puzzle.

        Args:
            puzzle_name: Name of the puzzle (e.g., "QuadraticRoot")
            language: Language ("javascript" or "python")

        Returns:
            The prompt string, or None if not found
        """
        self._load_prompts()
        key = f"{language}/{puzzle_name}"
        puzzle_data = self._prompts.get(key)
        if not puzzle_data:
            return None

        # Construct the prompt from the puzzle data
        docstring = puzzle_data.get("docstring", "")
        sat = puzzle_data.get("sat", "")
        sol = puzzle_data.get("sol", "")
        ans_type = puzzle_data.get("ans_type", "")

        prompt = f"""Solve the following {language} programming puzzle. Your task is to implement the sol function, to make it return a value that makes the sat function return True.

# {puzzle_name}

{docstring}

## Sat function

```{"javascript" if language == "javascript" else "python"}
{sat}
```

## Answer return value type

{ans_type}

## Sol header

```{"javascript" if language == "javascript" else "python"}
{sol}
```"""

        return prompt

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


def get_puzzle_prompt(puzzle_name: str, language: str = "javascript") -> Optional[str]:
    """
    Get the prompt for a specific puzzle.

    Args:
        puzzle_name: Name of the puzzle (e.g., "QuadraticRoot")
        language: Language ("javascript" or "python")

    Returns:
        The prompt string, or None if not found
    """
    return _puzzle_prompts.get_puzzle_prompt(puzzle_name, language)


def get_available_puzzles(language: str = "javascript") -> list:
    """
    Get list of available puzzles for a language.

    Args:
        language: Language ("javascript" or "python")

    Returns:
        List of puzzle names
    """
    return _puzzle_prompts.get_available_puzzles(language)
