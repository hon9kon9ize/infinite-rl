import sys
import json
import traceback
import os
import inspect
import importlib

# Add infinite_rl to path for imports
sys.path.append(os.path.dirname(__file__))

# Import PuzzleGenerator for type checking
from python_puzzles.puzzle_generator import PuzzleGenerator

# Dynamically import all puzzle classes
puzzles = {}
gen_pkg = importlib.import_module("python_puzzles.generators")
for module_name in dir(gen_pkg):
    if not module_name.startswith("_"):
        try:
            mod = importlib.import_module(f"python_puzzles.generators.{module_name}")
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


def evalPuzzle(puzzle, code, inputs):
    try:
        # Exec the user's code to define sol
        exec(code, globals())

        # Call sol with inputs
        result = sol(inputs)

        # Check against sat
        isCorrect = False
        if puzzle in puzzles:
            isCorrect = puzzles[puzzle].sat(result, *inputs.values())
        else:
            return {"error": f"Unknown puzzle: {puzzle}"}

        return {"result": result, "isCorrect": isCorrect}
    except Exception as err:
        return {"error": str(err), "stack": traceback.format_exc()}


# Read input from stdin
input_data = sys.stdin.read()
data = json.loads(input_data)
puzzle = data["puzzle"]
code = data["code"]
inputs = data["inputs"]

result = evalPuzzle(puzzle, code, inputs)
print(json.dumps(result))

sys.stdout.flush()
