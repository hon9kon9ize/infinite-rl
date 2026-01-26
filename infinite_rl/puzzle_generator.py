import json
import random
import inspect
from typing import Dict, List, Any
from .prompts import SYNTHESIS_SYSTEM_PROMPT, TASK_SYSTEM_PROMPTS, TYPE_PROMPTS

# Add current directory to path for imports like runner.py does
import sys
import os

sys.path.append(os.path.dirname(__file__))


class PuzzleDatasetGenerator:
    def __init__(self):
        pass

    def available_puzzles(self) -> List[str]:
        """Get list of available puzzle names."""
        # Import all puzzle generators dynamically like runner.py does
        import importlib
        import inspect
        from python_puzzles.puzzle_generator import PuzzleGenerator

        puzzles = {}
        # Import the generators package
        from python_puzzles import generators as gen_pkg

        for module_name in dir(gen_pkg):
            if not module_name.startswith("_"):
                try:
                    mod = importlib.import_module(
                        f"python_puzzles.generators.{module_name}"
                    )
                    for name, obj in inspect.getmembers(mod):
                        if (
                            inspect.isclass(obj)
                            and issubclass(obj, PuzzleGenerator)
                            and obj != PuzzleGenerator
                        ):
                            puzzles[obj.__name__] = obj
                except ImportError:
                    # Skip modules that can't be imported
                    pass

        return list(puzzles.keys())

    def load_puzzle(self, puzzle_name: str) -> Dict[str, Any]:
        """Load a specific puzzle by name."""
        # Import all puzzle generators dynamically
        import importlib
        import inspect
        from python_puzzles.puzzle_generator import PuzzleGenerator

        puzzles = {}
        gen_pkg = importlib.import_module("python_puzzles.generators")
        for module_name in dir(gen_pkg):
            if not module_name.startswith("_"):
                try:
                    mod = importlib.import_module(
                        f"python_puzzles.generators.{module_name}"
                    )
                    for name, obj in inspect.getmembers(mod):
                        if (
                            inspect.isclass(obj)
                            and issubclass(obj, PuzzleGenerator)
                            and obj != PuzzleGenerator
                        ):
                            puzzles[obj.__name__] = obj
                except ImportError:
                    # Skip modules that can't be imported
                    pass

        if puzzle_name not in puzzles:
            raise ValueError(f"Puzzle '{puzzle_name}' not found")

        puzzle_class = puzzles[puzzle_name]

        # Get the language (assume python for now, could be extended)
        language = "python"

        # Get examples from the puzzle instance
        examples = [puzzle_class.get_example()]

        return {
            "language": language,
            "examples": examples,
        }

    def generate_puzzle_sample(self, puzzle_name: str) -> Dict[str, Any]:
        """Generate a sample for a puzzle."""
        # Import all puzzle generators dynamically
        import importlib
        import inspect
        from python_puzzles.puzzle_generator import PuzzleGenerator

        puzzles = {}
        # Import the generators package
        from python_puzzles import generators as gen_pkg

        for module_name in dir(gen_pkg):
            if not module_name.startswith("_"):
                try:
                    mod = importlib.import_module(
                        f"python_puzzles.generators.{module_name}"
                    )
                    for name, obj in inspect.getmembers(mod):
                        if (
                            inspect.isclass(obj)
                            and issubclass(obj, PuzzleGenerator)
                            and obj != PuzzleGenerator
                        ):
                            puzzles[obj.__name__] = obj
                except ImportError:
                    # Skip modules that can't be imported
                    pass

        if puzzle_name not in puzzles:
            raise ValueError(f"Puzzle '{puzzle_name}' not found")

        puzzle_class = puzzles[puzzle_name]

        # Instantiate to get attributes
        puzzle_instance = puzzle_class()
        docstring = puzzle_instance.docstring
        sat_src = puzzle_instance.sat_src
        ans_type = puzzle_instance.ans_type
        arg_names = puzzle_instance.arg_names

        # Generate sol header
        if arg_names and len(arg_names) > 1:
            sol_args = ", ".join(arg_names[1:])
            sol_header = f"def sol({sol_args}):"
        else:
            sol_header = "def sol():"

        puzzle_info = self.load_puzzle(puzzle_name)
        inputs = random.choice(puzzle_info["examples"])

        # Format the prompt according to PROMPT.md
        prompt = f"""Solve the following python programming puzzle. Your task is continue implement the sol function, to make it returns a value that makes the sat function return True.

# {puzzle_name}

{docstring}

## Sat function

```python
{sat_src}
```

## Answer return value type

{ans_type}

## Sol header

```python
{sol_header}
```"""

        # Try to get the correct answer and sol source
        answer = None
        sol_src = None
        try:
            # Look for sol methods
            sol_methods = [
                method for method in dir(puzzle_class) if method.startswith("sol")
            ]
            if sol_methods:
                # Use the first sol method
                sol_method = getattr(puzzle_class, sol_methods[0])
                if callable(sol_method):
                    # Get the source code
                    try:
                        sol_src = inspect.getsource(sol_method)
                        # Remove the @staticmethod decorator if present
                        sol_src = sol_src.strip()
                        if sol_src.startswith("@staticmethod\n"):
                            sol_src = sol_src[len("@staticmethod\n") :].strip()
                    except:
                        sol_src = None
                    # Call the sol method with the inputs
                    if isinstance(inputs, dict):
                        answer = sol_method(**inputs)
                    else:
                        answer = sol_method(inputs)
        except Exception as e:
            # If we can't get the answer, that's okay - we'll just not provide it
            pass
        return {
            "prompt": prompt,
            "answer": json.dumps(
                {
                    "puzzle": puzzle_name,
                    "inputs": inputs,
                    "language": puzzle_info["language"],
                }
            ),
            "task": "puzzle",
            "puzzle": puzzle_name,
            "inputs": inputs,
            "language": puzzle_info["language"],
        }

    def generate_dataset(self, num_samples: int) -> List[Dict[str, Any]]:
        """Generate a dataset of samples."""
        # For now, only puzzle tasks
        samples = []
        puzzle_names = self.available_puzzles()

        for _ in range(num_samples):
            puzzle_name = random.choice(puzzle_names)
            sample = self.generate_puzzle_sample(puzzle_name)
            samples.append(sample)

        return samples
