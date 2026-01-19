from typing import Dict, List, Optional, Any
import multiprocessing as mp
import threading
from concurrent.futures import ProcessPoolExecutor
from typing import List, Dict, Any

from .reward_functions import get_reward_functions
from .reward_functions.reward_function import RewardFunctionScore


# Global worker variable to keep the orchestrator alive in the process
_worker_orch = None


class RewardOrchestrator:
    """Utility to orchestrate and run multiple reward functions.

    Features:
    - Load available reward functions via `get_reward_functions`
    - Optionally include lightweight repetition/length utilities as pseudo-reward functions
    - Compute an individual reward or multiple rewards for an example
    - Batch compute for lists of examples

    Examples:
        orch = RewardOrchestrator(timeout=5, include_repetition=True, include_length=True)
        score = orch.compute(model_output, expected_output, task="math")

    """

    def __init__(
        self,
        timeout: int = 10,
        gatekeeping: bool = False,
        include_repetition: bool = False,
        include_length: bool = False,
        repetition_n: int = 3,
        repetition_weight: float = -0.1,
        length_min_len: int = 1,
        length_max_len: int = 1000,
        length_target_len: Optional[int] = None,
        aux_score_weights: Dict[str, float] = {
            "lang_consistency": 0.2,
            "length": 0.1,
            "repetition": 0.1,
        },
    ):
        self.timeout = timeout
        self.gatekeeping = gatekeeping
        self.aux_score_weights = aux_score_weights
        self.fns = get_reward_functions(timeout=timeout)

        # Conditionally register auxiliary reward functions
        if include_repetition:
            # import here to avoid circular imports at module import time
            from .reward_functions.repetition import RepetitionRewardFunction

            self.fns["repetition"] = RepetitionRewardFunction(
                "repetition", timeout=timeout, n=repetition_n, weight=repetition_weight
            )

        if include_length:
            from .reward_functions.length import LengthRewardFunction

            self.fns["length"] = LengthRewardFunction(
                "length",
                timeout=timeout,
                min_len=length_min_len,
                max_len=length_max_len,
                target_len=length_target_len,
            )

    def available(self) -> List[str]:
        """Return list of available reward function keys."""
        return list(self.fns.keys())

    def get_fn(self, name: str):
        fn = self.fns.get(name)
        if fn is None:
            raise KeyError(f"Unknown reward function: {name}")
        if not getattr(fn, "initialized", False):
            fn.initialize()
        return fn

    def compute(
        self,
        model_output: str,
        expected_output: Any,
        task: str,
        lang: Optional[str] = None,
    ) -> RewardFunctionScore:
        """Compute the reward for a single task name (e.g., 'math', 'coding').

        Backwards compatible: if `lang` is None, returns a single `RewardFunctionScore` for `task`.

        If `lang` is provided, returns a dict of `{task_name: RewardFunctionScore}` that includes
        the main task and any auxiliary rewards that were registered at init time
        (for example, `lang_consistency`, `length`, or `repetition`).

        If `debug=True`, returns a dict with detailed internals for diagnosis instead
        of the aggregated `RewardFunctionScore`.
        """
        fn = self.get_fn(task)
        main_score = fn.compute_reward(model_output, expected_output)

        # Gatekeeping: if enabled, block auxiliaries and zero out all scores unless
        # the main task has a correctness score above 0.5
        if self.gatekeeping and main_score.correctness_score <= 0.5:
            zeroed = RewardFunctionScore(
                format_score=0.0,
                correctness_score=0.0,
                aux_score=0.0,
                error_msg={
                    "gatekeeping": "Gatekeeping active: main task correctness <= 0.5"
                },
            )
            return zeroed

        aux_results = {}

        if lang is not None:
            lang_fn = self.get_fn("lang_consistency")
            aux_results["lang_consistency"] = lang_fn.compute_reward(model_output, lang)

        # Include auxiliary length/repetition signals if those reward functions were registered
        length_fn = self.get_fn("length")
        # Pass main correctness into the length reward: treat >0.5 as correct
        is_main_correct = bool(main_score.correctness_score > 0.5)
        aux_results["length"] = length_fn.compute_reward(
            model_output, None, is_correct=is_main_correct
        )

        rep_fn = self.get_fn("repetition")
        aux_results["repetition"] = rep_fn.compute_reward(model_output, None)

        # Aggregate the rewards using configured weights (weighted average)
        aux_score = 0.0
        error_msg = main_score.error_msg
        for k, v in aux_results.items():
            w = (
                float(self.aux_score_weights.get(k, 1.0))
                if self.aux_score_weights
                else 1.0
            )
            if w < 0:
                raise ValueError("aux_score_weights must be non-negative")
            aux_score += float(v.aux_score) * w
            error_msg.update(v.error_msg)

        aggregated = RewardFunctionScore(
            format_score=main_score.format_score,
            correctness_score=main_score.correctness_score,
            aux_score=aux_score,
            error_msg=error_msg,
        )

        # Aggregated result: zero out format (aux-only context) and keep main correctness
        return aggregated


def _init_worker(orch_params: Dict[str, Any], tasks_to_prewarm: Optional[list] = None):
    """Initialize the orchestrator once per CPU core.

    Optionally prewarm only the reward functions that will actually be used in
    the current batch (reduces heavy startup cost when many reward functions
    might be available but only a few are needed).
    """
    global _worker_orch
    # Import the orchestrator implementation (absolute/relative imports both work in workers)
    from .reward_orchestrator import RewardOrchestrator

    _worker_orch = RewardOrchestrator(**orch_params)

    # If callers provide a list of tasks to prewarm, only initialize those.
    # This avoids expensive WASM/module warm-up for reward functions that are
    # not needed for the batch and significantly reduces worker startup time.
    if tasks_to_prewarm:
        for name in tasks_to_prewarm:
            try:
                _worker_orch.get_fn(name)
            except Exception:
                # Ignore initialization errors during warm-up to keep worker alive
                pass


def _process_sample(sample_data: Dict[str, Any]):
    """The function executed by the worker.

    If the sample dict contains a special key `_debug=True`, the worker will
    return the detailed debug dict produced by `RewardOrchestrator.compute(debug=True)`
    so that callers can inspect which functions ran and their raw scores.
    """
    global _worker_orch
    if sample_data.get("_debug"):
        return _worker_orch.compute(
            model_output=sample_data["model_output"],
            expected_output=sample_data["expected_output"],
            task=sample_data["task"],
            lang=sample_data.get("lang"),
            debug=True,
        )

    return _worker_orch.compute(
        model_output=sample_data["model_output"],
        expected_output=sample_data["expected_output"],
        task=sample_data["task"],
        lang=sample_data.get("lang"),
    )


class BatchRewardOrchestrator:
    def __init__(self, num_workers: int = None, **orch_params):
        self.num_workers = num_workers or mp.cpu_count()
        self.orch_params = orch_params

        # Persistent pool state
        self._pool: Optional[ProcessPoolExecutor] = None
        self._pool_tasks: set = set()
        self._pool_lock = threading.Lock()

    def _create_pool(self, tasks_to_prewarm: Optional[List[str]] = None):
        """Create or recreate the persistent worker pool, prewarming the given tasks."""
        # Shutdown existing pool if present
        if self._pool is not None:
            try:
                self._pool.shutdown(wait=True)
            except Exception:
                pass
            self._pool = None
            self._pool_tasks = set()

        # Create a new pool with the tasks to prewarm
        tasks_to_prewarm = sorted(tasks_to_prewarm or [])
        self._pool = ProcessPoolExecutor(
            max_workers=self.num_workers,
            initializer=_init_worker,
            initargs=(self.orch_params, tasks_to_prewarm),
        )
        self._pool_tasks = set(tasks_to_prewarm)

    def close(self):
        """Shutdown the persistent worker pool if it exists."""
        with self._pool_lock:
            if self._pool is not None:
                try:
                    self._pool.shutdown(wait=True)
                except Exception:
                    pass
                self._pool = None
                self._pool_tasks = set()

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass

    def compute_batch(self, samples: List[Dict[str, Any]]) -> List[Any]:
        """
        Runs rewards in parallel across CPU cores using a persistent worker pool.

        Input: List of dicts with keys ['model_output', 'expected_output', 'task', 'lang']

        If the process pool cannot be created (common when running code from stdin
        or certain interactive environments on macOS where the spawn start method
        requires an importable main module), we gracefully fall back to a
        sequential, in-process computation so callers still receive results.
        """
        # Determine which task types are present in the batch so we can prewarm
        tasks_in_batch = sorted({s.get("task") for s in samples if s.get("task")})

        # Try to use or create a persistent pool
        try:
            with self._pool_lock:
                if self._pool is None:
                    self._create_pool(tasks_in_batch)
                elif not set(tasks_in_batch).issubset(self._pool_tasks):
                    # Recreate pool with union of tasks to ensure needed prewarm
                    new_tasks = sorted(self._pool_tasks.union(tasks_in_batch))
                    self._create_pool(new_tasks)
                pool = self._pool

            results = list(pool.map(_process_sample, samples))
            return results
        except Exception as e:
            # If pool usage fails, discard it and fall back to sequential execution
            with self._pool_lock:
                if self._pool is not None:
                    try:
                        self._pool.shutdown(wait=False)
                    except Exception:
                        pass
                    self._pool = None
                    self._pool_tasks = set()

            import warnings

            warnings.warn(
                f"Process-based execution unavailable ({e!r}); falling back to sequential compute.",
                RuntimeWarning,
            )

            # Use a local orchestrator instance to compute samples one-by-one
            local_orch = RewardOrchestrator(**self.orch_params)
            results = []
            for s in samples:
                if s.get("_debug"):
                    results.append(
                        local_orch.compute(
                            model_output=s["model_output"],
                            expected_output=s["expected_output"],
                            task=s["task"],
                            lang=s.get("lang"),
                            debug=True,
                        )
                    )
                else:
                    results.append(
                        local_orch.compute(
                            model_output=s["model_output"],
                            expected_output=s["expected_output"],
                            task=s["task"],
                            lang=s.get("lang"),
                        )
                    )
            return results
