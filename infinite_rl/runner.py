import sys
import json
import traceback
import os
import inspect
import importlib

# Import PuzzleGenerator for type checking
from .python_puzzles.puzzle_generator import PuzzleGenerator
from .executor import Executor

# Dynamically import all puzzle classes
puzzles = {}
gen_pkg = importlib.import_module("infinite_rl.python_puzzles.generators")
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

# Initialize executor for JavaScript
executor = Executor()


def evalPuzzle(puzzle, code, inputs, language="python"):
    try:
        # Check if puzzle exists first
        if language.lower() != "javascript" and puzzle not in puzzles:
            return {"error": f"Unknown puzzle: {puzzle}"}

        result = None

        if language.lower() == "javascript":
            # Execute JavaScript code using WASM
            puzzle_input = json.dumps(
                {
                    "puzzle": puzzle,
                    "code": code,
                    "inputs": inputs,
                }
            )
            stdout, stderr = executor.run_single(puzzle_input, "javascript")
            if stderr:
                return {"error": stderr}
            try:
                js_result = json.loads(stdout)
                if "error" in js_result:
                    return js_result
                result = js_result["result"]
            except json.JSONDecodeError:
                return {"error": f"Invalid JSON output from JavaScript: {stdout}"}
        else:
            # Execute Python code
            # Exec the user's code to define sol
            exec(code, globals())

            # Call sol with inputs unpacked
            result = sol(*inputs.values())

        # Check against Python sat function
        isCorrect = False
        if puzzle in puzzles:
            isCorrect = puzzles[puzzle].sat(result, *inputs.values())
        else:
            return {"error": f"Unknown puzzle: {puzzle}"}

        return {"result": result, "isCorrect": isCorrect}
    except Exception as err:
        return {"error": str(err), "stack": traceback.format_exc()}


if __name__ == "__main__":
    # Read input from stdin
    input_data = sys.stdin.read()
    data = json.loads(input_data)
    puzzle = data["puzzle"]
    code = data["code"]
    inputs = data["inputs"]
    language = data.get("language", "python")

    result = evalPuzzle(puzzle, code, inputs, language)
    print(json.dumps(result))

    sys.stdout.flush()
