import json
import re
import subprocess
import os
import sys
import select
import threading
import queue
import atexit
from typing import TYPE_CHECKING, List, Optional

from .reward_function import RewardFunction, RewardFunctionScore

if TYPE_CHECKING:
    from ..task import Task


class _PersistentEvalPool:
    """Pool of long-running runner.py subprocesses for fast puzzle evaluation.

    Eliminates the ~200 ms Python interpreter startup cost of one-shot subprocesses
    by keeping worker processes alive between evaluations.  Workers communicate via
    stdin/stdout: one newline-terminated JSON object per request/response.

    Thread-safe: multiple threads may call evaluate() concurrently; each call
    checks out one worker from a queue, uses it exclusively, then returns it.
    """

    def __init__(self, n_workers: int) -> None:
        self._runner_path = os.path.join(
            os.path.dirname(__file__), "..", "runner.py"
        )
        self._available: queue.Queue = queue.Queue()
        for _ in range(n_workers):
            self._available.put(self._start_worker())
        atexit.register(self.shutdown)

    def _start_worker(self) -> subprocess.Popen:
        return subprocess.Popen(
            [sys.executable, "-u", self._runner_path, "--persistent"],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            # The protocol only uses stdout. Do not pipe stderr unless it is
            # drained; candidate code can otherwise fill the pipe and block.
            stderr=subprocess.DEVNULL,
            text=True,
        )

    def evaluate(self, data: dict, timeout: int) -> dict:
        """Send one evaluation request to a worker and return the parsed result.

        Checks out an idle worker, sends the request, waits (with timeout) for
        the response, then returns the worker to the pool.  On any failure the
        dead worker is replaced by a fresh one and an error dict is returned.
        """
        timeout = 10 if timeout is None else timeout
        worker = self._available.get()
        try:
            worker.stdin.write(json.dumps(data) + "\n")
            worker.stdin.flush()

            # select() gives us a non-blocking timeout without extra threads
            ready, _, _ = select.select([worker.stdout], [], [], timeout)
            if not ready:
                worker.kill()
                raise TimeoutError(f"worker timed out after {timeout}s")

            line = worker.stdout.readline()
            if not line:
                raise RuntimeError("worker process exited unexpectedly")

            result = json.loads(line.strip())
        except Exception as e:
            # Worker is broken — kill it and spawn a replacement so the pool
            # stays at full capacity.
            try:
                worker.kill()
                worker.wait(timeout=1)
            except Exception:
                pass
            self._available.put(self._start_worker())
            return {"error": str(e)}
        else:
            self._available.put(worker)
            return result

    def shutdown(self) -> None:
        while not self._available.empty():
            try:
                w = self._available.get_nowait()
                w.stdin.close()
                w.wait(timeout=2)
            except Exception:
                try:
                    w.kill()
                except Exception:
                    pass


# ---------------------------------------------------------------------------
# Module-level singleton — shared across all PuzzleRewardFunction instances
# on the same process.  Initialized lazily on first use.
# ---------------------------------------------------------------------------
_pool_instance: Optional[_PersistentEvalPool] = None
_pool_init_lock = threading.Lock()
_POOL_WORKERS = 16   # generous default; covers num_generations up to 16


def _get_pool() -> _PersistentEvalPool:
    global _pool_instance
    if _pool_instance is not None:
        return _pool_instance
    with _pool_init_lock:
        if _pool_instance is None:
            _pool_instance = _PersistentEvalPool(n_workers=_POOL_WORKERS)
    return _pool_instance


class PuzzleRewardFunction(RewardFunction):
    """Reward function for evaluating LLM-generated sol functions against programming puzzles.

    The expected_output should be a dict with 'puzzle', 'inputs', and optionally 'language'.

    Parallel batch evaluation: call compute_rewards_batch(tasks) to evaluate all
    completions concurrently. JavaScript evaluations reuse a module-level
    persistent subprocess pool; Python evaluations use fresh runner processes
    for isolation.
    """

    def __init__(
        self,
        task_name: str = "puzzle",
        language: str = "python",
        timeout: int = 5,
        answer_tag: str = "answer",
        think_tag: str = "think",
    ):
        super().__init__(
            task_name, timeout=timeout, answer_tag=answer_tag, think_tag=think_tag
        )
        self.language = language.lower()  # e.g., 'python' or 'javascript'

    def initialize(self):
        """Initialize the reward function."""
        self.initialized = True

    def _evaluate_one_shot(self, data: dict) -> dict:
        """Evaluate through runner.py in a fresh process.

        Python submissions are untrusted and can mutate imported modules. A
        fresh runner process keeps those mutations from leaking across
        evaluations while communicate() drains stdout/stderr to avoid pipe
        backpressure.
        """
        runner_path = os.path.join(os.path.dirname(__file__), "..", "runner.py")
        process = subprocess.Popen(
            [sys.executable, "-u", runner_path],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )

        runner_timeout = None if self.timeout is None else self.timeout + 1
        try:
            stdout, stderr = process.communicate(
                input=json.dumps(data), timeout=runner_timeout
            )
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=1)
            return {"error": f"Execution timed out after {self.timeout}s"}

        if process.returncode != 0:
            return {
                "error": f"Execution error (exit code {process.returncode}): {stderr}"
            }

        if not stdout or not stdout.strip():
            return {"error": "No output from execution"}

        return json.loads(stdout.strip())

    def compute_reward(
        self,
        task: "Task",
        **kwargs,
    ) -> RewardFunctionScore:
        if not self.initialized:
            self.initialize()

        # Parse expected_output from task
        expected_output = task.expected_answer
        if isinstance(expected_output, str):
            try:
                expected_output = json.loads(expected_output)
            except Exception:
                return RewardFunctionScore(
                    score=0.0, info="Invalid expected_output format"
                )

        puzzle_name = expected_output.get("puzzle")
        inputs = expected_output.get("inputs", {})
        language = expected_output.get("language", self.language)

        # Special case for simulation dummy puzzle
        if puzzle_name == "dummy_puzzle":
            import re as re_module

            lang_pattern = r"```(?:javascript)\b\s*(.*?)```"
            search_region = task.model_output
            think_close = f"</{self.think_tag}>"
            close_idx = task.model_output.find(think_close)
            if close_idx >= 0:
                search_region = task.model_output[close_idx + len(think_close):]

            match = re_module.search(
                lang_pattern, search_region, re_module.DOTALL | re_module.IGNORECASE
            )
            if not match:
                if close_idx >= 0:
                    match = re_module.search(
                        lang_pattern, task.model_output, re_module.DOTALL | re_module.IGNORECASE
                    )
            if not match:
                return RewardFunctionScore(
                    score=0.0,
                    info=f"Missing code block with language 'javascript' inside <{self.think_tag}> tags.",
                )
            code = match.group(1).strip()
            if "return true" in code:
                return RewardFunctionScore(
                    score=1.0,
                    info="Dummy puzzle simulation - correct",
                )
            else:
                return RewardFunctionScore(
                    score=0.0,
                    info="Dummy puzzle simulation - incorrect",
                )

        # Extract code block (```language ... ```)
        # Prefer code after </think> to avoid echoing prompt content.
        import re as re_module

        lang_pattern = rf"```(?:{language})\b\s*(.*?)```"
        search_region = task.model_output
        think_close = f"</{self.think_tag}>"
        close_idx = task.model_output.find(think_close)
        if close_idx >= 0:
            search_region = task.model_output[close_idx + len(think_close):]

        match = re_module.search(
            lang_pattern, search_region, re_module.DOTALL | re_module.IGNORECASE
        )

        if not match:
            if close_idx >= 0:
                match = re_module.search(
                    lang_pattern, task.model_output, re_module.DOTALL | re_module.IGNORECASE
                )
                if not match:
                    return RewardFunctionScore(
                        score=0.0,
                        info=f"Missing code block with language '{language}' in response.",
                    )
            else:
                return RewardFunctionScore(
                    score=0.0,
                    info=f"Missing code block with language '{language}' in response.",
                )

        code_content = match.group(1).strip()

        sol_pattern = r"def sol\s*\(|function sol\s*\("
        if not re.search(sol_pattern, code_content):
            return RewardFunctionScore(
                score=0.0, info="Code must define a sol function"
            )

        request = {
            "puzzle": puzzle_name,
            "inputs": inputs,
            "code": code_content,
            "language": language,
            "timeout": self.timeout,
        }

        # JavaScript runs inside the WASM executor, so the persistent worker is
        # safe to reuse. Python candidate code runs in a fresh runner process to
        # avoid leaking user mutations across evaluations.
        try:
            if language.lower() == "python":
                output = self._evaluate_one_shot(request)
            else:
                output = _get_pool().evaluate(request, timeout=self.timeout)

            if "error" in output:
                error_msg = output["error"]
                if "stack" in output and output["stack"]:
                    error_msg += f"\nStack trace:\n{output['stack']}"
                return RewardFunctionScore(
                    score=0.0,
                    info=f"Evaluation error: {error_msg}",
                )
            is_correct = output.get("isCorrect", False)
            if is_correct:
                return RewardFunctionScore(score=1.0)
            else:
                return RewardFunctionScore(
                    score=0.0,
                    info="Puzzle check failed",
                )
        except Exception as e:
            return RewardFunctionScore(score=0.0, info=f"Evaluation failed: {str(e)}")

    def compute_rewards_batch(self, tasks: List["Task"]) -> List[RewardFunctionScore]:
        """Evaluate multiple completions concurrently.

        Each task must have its own model_output already set (use shallow copies
        from curriculum.compute_rewards).  All N evaluations are submitted to the
        worker path at once via a ThreadPoolExecutor; wall-clock time is roughly
        that of a single evaluation rather than N serial evaluations.

        Args:
            tasks: List of Task objects (same puzzle, different model_output).

        Returns:
            List of RewardFunctionScore, one per task, in input order.
        """
        from concurrent.futures import ThreadPoolExecutor

        if not tasks:
            return []
        if not self.initialized:
            self.initialize()
        with ThreadPoolExecutor(max_workers=len(tasks)) as ex:
            return list(ex.map(self.compute_reward, tasks))
