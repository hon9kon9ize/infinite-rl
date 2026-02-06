import sys
import json
import traceback
import os
import inspect
import importlib

# Handle both script execution and module import
# When run as script, __package__ is None; when imported as module, it's set
if __package__ is None or __package__ == "":
    # Running as script - add parent directory to path and use absolute imports
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from infinite_rl.python_puzzles.puzzle_generator import PuzzleGenerator
    from infinite_rl.executor import Executor

    base_module = "infinite_rl.python_puzzles.generators"
else:
    # Running as module - use relative imports
    from .python_puzzles.puzzle_generator import PuzzleGenerator
    from .executor import Executor

    base_module = "infinite_rl.python_puzzles.generators"

# Lazy-load puzzle generators only when needed (reduces startup time by ~200-400ms)
puzzles = {}
_puzzle_loader_cache = {}


def _get_puzzle_class(puzzle_name):
    """Lazy-load a puzzle class only when needed."""
    if puzzle_name in _puzzle_loader_cache:
        return _puzzle_loader_cache[puzzle_name]

    # Search for puzzle class in generators
    gen_pkg = importlib.import_module(base_module)
    for module_name in dir(gen_pkg):
        if not module_name.startswith("_"):
            try:
                mod = importlib.import_module(
                    f"infinite_rl.python_puzzles.generators.{module_name}"
                )
                for name, obj in inspect.getmembers(mod):
                    if (
                        inspect.isclass(obj)
                        and issubclass(obj, PuzzleGenerator)
                        and obj != PuzzleGenerator
                        and obj.__name__ == puzzle_name
                    ):
                        _puzzle_loader_cache[puzzle_name] = obj
                        return obj
            except ImportError:
                pass
    return None


# Initialize executor for JavaScript (lazy - only when needed)
executor = None


def _get_executor():
    """Lazy-initialize executor only when needed for JavaScript."""
    global executor
    if executor is None:
        executor = Executor()
    return executor


def evalPuzzle(puzzle, code, inputs, language="python"):
    try:
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
            exec_instance = _get_executor()
            stdout, stderr = exec_instance.run_single(puzzle_input, "javascript")
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

        # Check against Python sat function - lazy load puzzle class
        puzzle_class = _get_puzzle_class(puzzle)
        if puzzle_class is None:
            return {"error": f"Unknown puzzle: {puzzle}"}

        isCorrect = puzzle_class.sat(result, *inputs.values())

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
