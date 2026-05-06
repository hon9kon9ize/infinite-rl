import sys
import json
import traceback
import os
import inspect
import importlib
import subprocess
import tempfile

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
_js_puzzle_info_cache = None


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


def _get_js_puzzle_info(puzzle_name):
    """Return JavaScript puzzle metadata from packaged runtimes/puzzles.json."""
    global _js_puzzle_info_cache
    if _js_puzzle_info_cache is None:
        try:
            if __package__ is None or __package__ == "":
                runtime_path = os.path.join(
                    os.path.dirname(os.path.abspath(__file__)),
                    "runtimes",
                    "puzzles.json",
                )
                with open(runtime_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
            else:
                from importlib import resources

                resource = resources.files("infinite_rl.runtimes").joinpath(
                    "puzzles.json"
                )
                with resource.open("r", encoding="utf-8") as f:
                    data = json.load(f)
            _js_puzzle_info_cache = data.get("javascript", {})
        except Exception:
            _js_puzzle_info_cache = {}
    return _js_puzzle_info_cache.get(puzzle_name)


_PYTHON_CHILD_CODE = r"""
import json
import sys
import traceback

_json_load = json.load
_json_dumps = json.dumps
_open = open
_format_exc = traceback.format_exc

payload_path = sys.argv[1]
result_path = sys.argv[2]


def _write_result(payload):
    try:
        encoded = _json_dumps(payload)
    except Exception as dump_err:
        encoded = _json_dumps({
            "error": f"Result is not JSON serializable: {dump_err}",
            "stack": _format_exc(),
        })
    with _open(result_path, "w", encoding="utf-8") as f:
        f.write(encoded)


try:
    with _open(payload_path, "r", encoding="utf-8") as f:
        data = _json_load(f)

    local_ns = {}
    exec(data["code"], local_ns)
    sol_fn = local_ns.get("sol")
    if sol_fn is None:
        _write_result({"error": "sol function not defined in submitted code"})
    else:
        result = sol_fn(*data.get("inputs", {}).values())
        _write_result({"result": result})
except SyntaxError as err:
    _write_result({"error": f"Syntax error: {err}", "stack": _format_exc()})
except Exception as err:
    _write_result({"error": str(err), "stack": _format_exc()})
"""


def _eval_python_solution(code, inputs, timeout=None):
    """Run submitted Python code in a child process and return its result.

    The parent runner performs SAT validation after the child exits, so user code
    cannot monkeypatch puzzle classes or leak module mutations into later
    persistent requests. stdout/stderr are drained with communicate().
    """
    payload_path = None
    result_path = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", delete=False, encoding="utf-8"
        ) as payload_file:
            json.dump({"code": code, "inputs": inputs}, payload_file)
            payload_path = payload_file.name

        result_fd, result_path = tempfile.mkstemp(
            prefix="infinite_rl_python_result_", suffix=".json"
        )
        os.close(result_fd)

        process = subprocess.Popen(
            [sys.executable, "-u", "-c", _PYTHON_CHILD_CODE, payload_path, result_path],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )

        try:
            _, stderr = process.communicate(timeout=timeout)
        except subprocess.TimeoutExpired:
            process.kill()
            process.communicate()
            return {"error": f"Execution timed out after {timeout}s"}

        if result_path and os.path.exists(result_path):
            with open(result_path, "r", encoding="utf-8") as result_file:
                raw_result = result_file.read().strip()
            if raw_result:
                return json.loads(raw_result)

        if process.returncode != 0:
            return {
                "error": f"Execution error (exit code {process.returncode}): {stderr}"
            }
        return {"error": "No output from Python execution"}
    except Exception as err:
        return {"error": str(err), "stack": traceback.format_exc()}
    finally:
        for path in (payload_path, result_path):
            if path and os.path.exists(path):
                try:
                    os.remove(path)
                except OSError:
                    pass


def evalPuzzle(puzzle, code, inputs, language="python", timeout=None):
    try:
        result = None

        if language.lower() == "javascript":
            js_puzzle_info = _get_js_puzzle_info(puzzle)
            js_sat = js_puzzle_info.get("sat") if js_puzzle_info else None

            # Execute JavaScript code using WASM
            request = {
                "puzzle": puzzle,
                "code": code,
                "inputs": inputs,
            }
            if js_sat:
                request["sat"] = js_sat
            puzzle_input = json.dumps(request)
            exec_instance = _get_executor()
            stdout, stderr = exec_instance.run_single(puzzle_input, "javascript")
            if stderr:
                return {"error": stderr}
            try:
                js_result = json.loads(stdout)
                if "error" in js_result:
                    return js_result
                if "isCorrect" in js_result:
                    return {
                        "result": js_result.get("result"),
                        "isCorrect": bool(js_result["isCorrect"]),
                    }
                if js_sat:
                    return {
                        "error": "JavaScript runtime did not validate SAT. Rebuild puzzle_js.wasm."
                    }
                result = js_result["result"]
            except json.JSONDecodeError:
                return {"error": f"Invalid JSON output from JavaScript: {stdout}"}
        else:
            child_result = _eval_python_solution(code, inputs, timeout=timeout)
            if "error" in child_result:
                return child_result
            result = child_result.get("result")

        # Check against Python sat function - lazy load puzzle class
        puzzle_class = _get_puzzle_class(puzzle)
        if puzzle_class is None:
            return {"error": f"Unknown puzzle: {puzzle}"}

        isCorrect = puzzle_class.sat(result, *inputs.values())

        return {"result": result, "isCorrect": isCorrect}
    except Exception as err:
        return {"error": str(err), "stack": traceback.format_exc()}


if __name__ == "__main__":
    _persistent = "--persistent" in sys.argv

    if _persistent:
        # Persistent mode: handle multiple evaluations without restarting the interpreter.
        # Parent sends one JSON object per line; we reply with one JSON object per line.
        # Loop exits when stdin is closed (parent sends EOF).
        while True:
            line = sys.stdin.readline()
            if not line:
                break
            line = line.strip()
            if not line:
                continue
            try:
                data = json.loads(line)
                result = evalPuzzle(
                    data["puzzle"],
                    data["code"],
                    data["inputs"],
                    data.get("language", "python"),
                    data.get("timeout"),
                )
            except Exception as e:
                result = {"error": str(e), "stack": traceback.format_exc()}
            print(json.dumps(result), flush=True)
    else:
        # Original one-shot mode (backward compatible)
        input_data = sys.stdin.read()
        data = json.loads(input_data)
        puzzle = data["puzzle"]
        code = data["code"]
        inputs = data["inputs"]
        language = data.get("language", "python")

        result = evalPuzzle(puzzle, code, inputs, language, data.get("timeout"))
        print(json.dumps(result))
        sys.stdout.flush()
