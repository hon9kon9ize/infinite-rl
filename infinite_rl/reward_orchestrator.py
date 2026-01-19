from typing import Dict, List, Optional, Any
import multiprocessing as mp
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
        """
        fn = self.get_fn(task)
        main_score = fn.compute_reward(model_output, expected_output)

        # Gatekeeping: if enabled, block auxiliaries and zero out all scores unless
        # the main task has a correctness score above 0.5
        if self.gatekeeping and main_score.correctness_score <= 0.5:
            return RewardFunctionScore(
                format_score=0.0, correctness_score=0.0, aux_score=0.0
            )

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
        for k, v in aux_results.items():
            w = (
                float(self.aux_score_weights.get(k, 1.0))
                if self.aux_score_weights
                else 1.0
            )
            if w < 0:
                raise ValueError("aux_score_weights must be non-negative")
            aux_score += float(v.aux_score) * w

        # Aggregated result: zero out format (aux-only context) and keep main correctness
        return RewardFunctionScore(
            format_score=main_score.format_score,
            correctness_score=main_score.correctness_score,
            aux_score=aux_score,
        )


def _init_worker(orch_params: Dict[str, Any]):
    """Initialize the orchestrator once per CPU core."""
    global _worker_orch
    # Import the orchestrator implementation (absolute/relative imports both work in workers)
    from .reward_orchestrator import RewardOrchestrator

    _worker_orch = RewardOrchestrator(**orch_params)
    # Warm up available functions. Wrap in try/except to avoid crashing a worker
    # during optional heavy initialization (WASM/runtime) which may not be required
    # for the set of rewards actually used in the batch.
    for name in _worker_orch.available():
        try:
            _worker_orch.get_fn(name)
        except Exception:
            # Ignore initialization errors during warm-up to keep worker alive
            pass


def _process_sample(sample_data: Dict[str, Any]):
    """The function executed by the worker."""
    global _worker_orch
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

    def compute_batch(self, samples: List[Dict[str, Any]]) -> List[Any]:
        """
        Runs rewards in parallel across CPU cores.
        Input: List of dicts with keys ['model_output', 'expected_output', 'task', 'lang']
        """
        with ProcessPoolExecutor(
            max_workers=self.num_workers,
            initializer=_init_worker,
            initargs=(self.orch_params,),
        ) as executor:
            results = list(executor.map(_process_sample, samples))
        return results
