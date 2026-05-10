"""
Simplified training script for fine-tuning Gemma-3-4B with TRL's GRPO trainer
using Infinite-RL's curriculum learning and reward functions.

This version is optimized for vLLM COLOCATE mode ONLY (simplified, no server mode).

NOTE: This script assumes the model vocabulary has already been expanded using train/model_expand.py
and special tokens have been added. It does NOT handle vocabulary expansion.

Integrates:
- TRL's GRPO (Group Relative Policy Optimization) trainer
- vLLM in colocate mode for efficient generation
- Infinite-RL's curriculum learning for progressive difficulty
- Custom reward functions from Infinite-RL for task evaluation
"""

import argparse
import atexit
import faulthandler
import json
import logging
import os
import signal
import sys
import threading
import time
from pathlib import Path
from typing import Optional, List, Callable
from dataclasses import dataclass, asdict

import torch
import numpy as np
import wandb
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainerCallback,
)
from trl import GRPOTrainer, GRPOConfig
from peft import LoraConfig, get_peft_model, TaskType

# Import infinite-rl components
from infinite_rl.curriculum import CurriculumLearning
from infinite_rl.dynamic_dataset import DynamicCurriculumDataset
import functools


logger = logging.getLogger("infinite_rl.train")
_FAULTHANDLER_FILES = []


def _get_env_value(*names: str, default: str = "unknown") -> str:
    """Return the first non-empty environment variable value."""
    for name in names:
        value = os.environ.get(name)
        if value:
            return value
    return default


class _RankLogFilter(logging.Filter):
    """Add distributed/HPC context fields to every log record."""

    def filter(self, record: logging.LogRecord) -> bool:
        record.rank = _get_env_value("RANK", "SLURM_PROCID", "LOCAL_RANK")
        record.local_rank = _get_env_value("LOCAL_RANK", "SLURM_LOCALID")
        record.world_size = _get_env_value("WORLD_SIZE", "SLURM_NTASKS")
        record.job_id = _get_env_value("SLURM_JOB_ID", "JOB_ID")
        return True


def setup_training_logging(
    output_dir: Path,
    train_log_file: Optional[str] = None,
    log_level: str = "INFO",
) -> Path:
    """Configure flushed file logging for HPC jobs."""
    output_dir.mkdir(parents=True, exist_ok=True)
    rank = _get_env_value("RANK", "SLURM_PROCID", default="")
    local_rank = _get_env_value("LOCAL_RANK", "SLURM_LOCALID", default="")

    if train_log_file:
        log_path = Path(train_log_file)
        if not log_path.is_absolute():
            log_path = output_dir / log_path
    else:
        rank_suffix = f".rank{rank}" if rank else ""
        local_suffix = f".local{local_rank}" if local_rank and local_rank != rank else ""
        log_path = output_dir / f"train{rank_suffix}{local_suffix}.log"

    log_path.parent.mkdir(parents=True, exist_ok=True)
    numeric_level = getattr(logging, log_level.upper(), logging.INFO)
    formatter = logging.Formatter(
        "%(asctime)s %(levelname)s "
        "[pid=%(process)d rank=%(rank)s local_rank=%(local_rank)s "
        "world=%(world_size)s job=%(job_id)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    rank_filter = _RankLogFilter()
    stream_handler = logging.StreamHandler(sys.stderr)
    stream_handler.setFormatter(formatter)
    stream_handler.addFilter(rank_filter)

    file_handler = logging.FileHandler(log_path, mode="a", encoding="utf-8")
    file_handler.setFormatter(formatter)
    file_handler.addFilter(rank_filter)

    logging.basicConfig(
        level=numeric_level,
        handlers=[stream_handler, file_handler],
        force=True,
    )
    logging.captureWarnings(True)
    _install_exception_logging()
    _install_faulthandler(log_path)

    logger.info("Logging initialized. train_log_file=%s", log_path)
    logger.info(
        "Runtime context: pid=%s rank=%s local_rank=%s world_size=%s slurm_job_id=%s cwd=%s",
        os.getpid(),
        _get_env_value("RANK", "SLURM_PROCID"),
        _get_env_value("LOCAL_RANK", "SLURM_LOCALID"),
        _get_env_value("WORLD_SIZE", "SLURM_NTASKS"),
        _get_env_value("SLURM_JOB_ID", "JOB_ID"),
        Path.cwd(),
    )
    return log_path


def _install_exception_logging() -> None:
    """Log uncaught exceptions from the main thread and background threads."""

    def log_excepthook(exc_type, exc_value, exc_traceback):
        logger.critical(
            "Uncaught exception",
            exc_info=(exc_type, exc_value, exc_traceback),
        )
        sys.__excepthook__(exc_type, exc_value, exc_traceback)

    sys.excepthook = log_excepthook

    if hasattr(threading, "excepthook"):
        original_threading_excepthook = threading.excepthook

        def log_thread_excepthook(args):
            logger.critical(
                "Uncaught exception in thread %s",
                args.thread.name if args.thread else "<unknown>",
                exc_info=(args.exc_type, args.exc_value, args.exc_traceback),
            )
            original_threading_excepthook(args)

        threading.excepthook = log_thread_excepthook


def _install_faulthandler(log_path: Path) -> None:
    """Enable low-level Python traceback dumps into the training log."""
    try:
        fh = open(log_path, "a", encoding="utf-8", buffering=1)
        _FAULTHANDLER_FILES.append(fh)
        faulthandler.enable(file=fh, all_threads=True)
        if hasattr(signal, "SIGUSR1"):
            faulthandler.register(signal.SIGUSR1, file=fh, all_threads=True)
            logger.info(
                "Faulthandler enabled. Send SIGUSR1 to pid=%s to dump all Python thread stacks.",
                os.getpid(),
            )
        atexit.register(_close_faulthandler_files)
    except Exception:
        logger.warning("Could not enable faulthandler logging", exc_info=True)


def _close_faulthandler_files() -> None:
    try:
        faulthandler.disable()
    except Exception:
        pass
    for fh in _FAULTHANDLER_FILES:
        try:
            fh.close()
        except Exception:
            pass


def _runtime_diagnostics() -> str:
    """Return compact CPU/GPU memory diagnostics for heartbeat logs."""
    parts = []
    try:
        import resource

        # macOS reports ru_maxrss in bytes; Linux reports KiB. HPC is almost
        # always Linux, but keep the value labelled to avoid false precision.
        parts.append(f"maxrss={resource.getrusage(resource.RUSAGE_SELF).ru_maxrss}")
    except Exception:
        pass

    try:
        if torch.cuda.is_available():
            device = torch.cuda.current_device()
            allocated_gb = torch.cuda.memory_allocated(device) / (1024**3)
            reserved_gb = torch.cuda.memory_reserved(device) / (1024**3)
            parts.append(
                f"cuda_device={device} cuda_allocated_gb={allocated_gb:.2f} "
                f"cuda_reserved_gb={reserved_gb:.2f}"
            )
        else:
            parts.append("cuda_available=False")
    except Exception as exc:
        parts.append(f"cuda_diagnostics_error={exc}")

    return " ".join(parts)


class TrainingProgressState:
    """Thread-safe state shared by callbacks, reward code, and heartbeat logs."""

    def __init__(self):
        self._lock = threading.Lock()
        self.phase = "starting"
        self.event = "init"
        self.trainer_step = 0
        self.curriculum_step = 0
        self.last_reward_task = None
        self.last_event_monotonic = time.monotonic()

    def update(self, **kwargs) -> None:
        with self._lock:
            for key, value in kwargs.items():
                setattr(self, key, value)
            self.last_event_monotonic = time.monotonic()

    def snapshot(self) -> dict:
        with self._lock:
            return {
                "phase": self.phase,
                "event": self.event,
                "trainer_step": self.trainer_step,
                "curriculum_step": self.curriculum_step,
                "last_reward_task": self.last_reward_task,
                "idle_seconds": time.monotonic() - self.last_event_monotonic,
            }


class TrainingHeartbeat:
    """Periodic alive/progress log for long HPC jobs."""

    def __init__(self, state: TrainingProgressState, interval_seconds: int):
        self.state = state
        self.interval_seconds = interval_seconds
        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None

    def start(self) -> None:
        if self.interval_seconds <= 0:
            logger.info("Heartbeat logging disabled")
            return
        self._thread = threading.Thread(
            target=self._run,
            name="training-heartbeat",
            daemon=True,
        )
        self._thread.start()
        logger.info("Heartbeat logging started. interval_seconds=%s", self.interval_seconds)

    def stop(self) -> None:
        self._stop_event.set()
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=5)
        logger.info("Heartbeat logging stopped")

    def _run(self) -> None:
        while not self._stop_event.wait(self.interval_seconds):
            snapshot = self.state.snapshot()
            logger.info(
                "heartbeat phase=%s event=%s trainer_step=%s curriculum_step=%s "
                "last_reward_task=%s idle_seconds=%.1f %s",
                snapshot["phase"],
                snapshot["event"],
                snapshot["trainer_step"],
                snapshot["curriculum_step"],
                snapshot["last_reward_task"],
                snapshot["idle_seconds"],
                _runtime_diagnostics(),
            )


def make_json_serializable(obj):
    """Convert non-JSON-serializable objects to strings recursively.

    Handles: torch.dtype, numpy.dtype, Path, and other common types.
    """
    if isinstance(obj, dict):
        return {k: make_json_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [make_json_serializable(item) for item in obj]
    elif isinstance(obj, (torch.dtype, np.dtype)):
        return str(obj)
    elif isinstance(obj, Path):
        return str(obj)
    elif isinstance(obj, type):
        return obj.__name__
    elif hasattr(obj, "__dict__") and not isinstance(
        obj, (str, int, float, bool, type(None))
    ):
        # For custom objects, try to convert to dict
        try:
            return make_json_serializable(obj.__dict__)
        except:
            return str(obj)
    else:
        return obj


def _summarize_scores(scores: List[float]) -> str:
    if not scores:
        return "count=0"
    numeric_scores = [float(score) for score in scores]
    preview = ", ".join(f"{score:.4f}" for score in numeric_scores[:8])
    if len(numeric_scores) > 8:
        preview += ", ..."
    return (
        f"count={len(numeric_scores)} min={min(numeric_scores):.4f} "
        f"max={max(numeric_scores):.4f} "
        f"avg={sum(numeric_scores) / len(numeric_scores):.4f} "
        f"values=[{preview}]"
    )


@dataclass
class InfiniteRLConfig:
    """Configuration for Infinite-RL curriculum integration."""

    # Curriculum parameters
    window_size: int = 80
    success_rate_threshold: float = 0.65
    variance_threshold: float = 0.15
    demote_threshold: float = 0.4
    warmup_step: int = 100
    level_change_cooldown: int = 10

    num_generations: int = 4
    pre_reasoning_dataset: Optional[str] = None
    pre_reasoning_split: str = "train"
    pre_reasoning_learning_rate: float = 0.0

    # Reward function parameters
    timeout: int = 40
    answer_tag: str = "answer"
    think_tag: str = "think"
    reasoning_language: str = "en"
    reasoning_template: bool = False
    use_system_prompt: bool = True

    # Auxiliary rewards
    use_format: bool = True
    use_reasoning_steps: bool = True
    use_response_content: bool = True
    use_lang_consistency: bool = True
    use_length: bool = True
    aux_weight: float = 0.5

    # LLM Judge configuration
    use_llm_judge: bool = False
    llm_judge_host: str = "localhost"
    llm_judge_port: int = 8000
    llm_judge_model: str = "Skywork/Skywork-Reward-V2-Qwen3-4B"
    llm_judge_weight: float = 0.1

    # Output
    log_file: Optional[str] = "curriculum_learning_log.jsonl"


def create_curriculum(config: InfiniteRLConfig) -> CurriculumLearning:
    """Initialize the curriculum learning system from Infinite-RL.

    Args:
        config: Infinite-RL configuration

    Returns:
        Initialized CurriculumLearning instance
    """
    # Prepare LLM Judge kwargs if enabled
    llm_judge_kwargs = {}
    if config.use_llm_judge:
        llm_judge_kwargs = {
            "api_host": config.llm_judge_host,
            "api_port": config.llm_judge_port,
            "model_name": config.llm_judge_model,
        }

    curriculum = CurriculumLearning(
        timeout=config.timeout,
        answer_tag=config.answer_tag,
        think_tag=config.think_tag,
        reasoning_language=config.reasoning_language,
        reasoning_template=config.reasoning_template,
        use_system_prompt=config.use_system_prompt,
        aux_weight=config.aux_weight,
        use_format=config.use_format,
        use_reasoning_steps=config.use_reasoning_steps,
        use_response_content=config.use_response_content,
        use_lang_consistency=config.use_lang_consistency,
        use_llm_judge=config.use_llm_judge,
        use_length=config.use_length,
        llm_judge_kwargs=llm_judge_kwargs,
        window_size=config.window_size,
        llm_judge_weight=config.llm_judge_weight,
        success_rate_threshold=config.success_rate_threshold,
        variance_threshold=config.variance_threshold,
        demote_threshold=config.demote_threshold,
        warmup_step=config.warmup_step,
        level_change_cooldown=config.level_change_cooldown,
        num_generations=config.num_generations,
        log_file=config.log_file,
        pre_reasoning_dataset=config.pre_reasoning_dataset,
        pre_reasoning_split=config.pre_reasoning_split,
        pre_reasoning_learning_rate=config.pre_reasoning_learning_rate,
    )
    return curriculum


def build_dynamic_dataset(
    curriculum: CurriculumLearning,
    num_samples: int = 10000,
) -> DynamicCurriculumDataset:
    """Build a dynamic dataset that generates prompts on-demand.

    This ensures each sample always reflects the current curriculum level
    without needing to pre-generate or refresh the dataset.

    Args:
        curriculum: Initialized CurriculumLearning instance
        num_samples: Virtual size of the dataset (for trainer iteration)

    Returns:
        DynamicCurriculumDataset instance
    """
    learning_stats = curriculum.get_learning_stats()
    logger.info(
        "Creating dynamic dataset. num_samples=%s current_level=%s num_generations=%s",
        num_samples,
        learning_stats.get("current_level"),
        curriculum.num_generations,
    )
    print(f"\n[DEBUG] Creating dynamic dataset:")
    print(f"  - Virtual size: {num_samples} samples")
    print(f"  - Prompts generated on-demand from curriculum")
    print(f"  - Current curriculum level: {learning_stats['current_level']}")

    return DynamicCurriculumDataset(curriculum, num_samples)


def create_curriculum_reward_func(
    curriculum: CurriculumLearning,
    progress_state: Optional[TrainingProgressState] = None,
    log_reward_batches: bool = True,
) -> Callable:
    """Create a reward function compatible with GRPOTrainer.

    The reward function uses curriculum.compute_reward() which:
    1. Computes primary + auxiliary rewards
    2. Accumulates generations
    3. When batch complete: computes LLM Judge, recomputes combined scores
    4. Returns the combined score directly

    Args:
        curriculum: Initialized CurriculumLearning instance

    Returns:
        A callable that computes rewards for completions
    """

    def reward_func(
        completions: List[str],
        **kwargs,
    ) -> List[float]:
        """Compute rewards for completions using Infinite-RL's curriculum.

        Args:
            completions: List of model completions
            **kwargs: Additional arguments from trainer, including task_metadata

        Returns:
            List of reward scores in range [0, 1] (combined scores with judge)
        """
        task_metadata = kwargs.get("task_metadata")
        from collections import defaultdict

        reward_batch_started = time.monotonic()
        if progress_state is not None:
            progress_state.update(
                phase="reward_batch",
                event="reward_batch_start",
                curriculum_step=curriculum.global_step,
            )
        if log_reward_batches:
            logger.info(
                "reward_batch_start completions=%s metadata_present=%s curriculum_step=%s",
                len(completions),
                bool(task_metadata),
                curriculum.global_step,
            )

        # CRITICAL: Preserve original completion order by storing (index, completion) pairs
        # This prevents rewards from being assigned to wrong generations
        indexed_completions = []  # List of (original_index, task_id, completion)
        grouped = defaultdict(list)  # task_id -> list of (original_index, completion)

        for i, completion in enumerate(completions):
            if task_metadata and i < len(task_metadata):
                task_id = task_metadata[i]["task_id"]
            else:
                # Fallback if no metadata
                task_id = f"task_{i // curriculum.num_generations}"

            indexed_completions.append((i, task_id, completion))
            grouped[task_id].append((i, completion))

        if log_reward_batches:
            logger.info(
                "reward_batch_grouped task_count=%s task_ids=%s",
                len(grouped),
                list(grouped.keys())[:20],
            )

        # Collect all rewards with their original indices
        reward_with_index = {}  # original_index -> reward_score

        # Process each task's completions in batch
        for task_id, index_completion_list in grouped.items():
            task_started = time.monotonic()
            try:
                # Extract content from TRL's chat format for all completions
                completion_texts = []
                indices = []
                for orig_idx, completion in index_completion_list:
                    indices.append(orig_idx)
                    completion_text = completion
                    if isinstance(completion, list) and len(completion) > 0:
                        # Extract from list of dicts format
                        completion_text = completion[0].get("content", "")
                    elif isinstance(completion, dict):
                        # Extract from single dict format
                        completion_text = completion.get("content", "")

                    # Ensure we have a string
                    if not isinstance(completion_text, str):
                        print(
                            f"Warning: Completion for task {task_id} is not a string after extraction. Type: {type(completion_text)}, value: {completion_text}"
                        )
                        completion_texts.append("")
                    else:
                        completion_texts.append(completion_text)

                output_lengths = [len(text) for text in completion_texts]
                task = curriculum.session.get_task(task_id)
                task_type = getattr(task, "task_type", "unknown")
                task_level = getattr(task, "level", "unknown")
                if progress_state is not None:
                    progress_state.update(
                        phase="reward_task",
                        event="reward_task_start",
                        curriculum_step=curriculum.global_step,
                        last_reward_task=task_id,
                    )
                if log_reward_batches:
                    logger.info(
                        "reward_task_start task_id=%s task_type=%s level=%s "
                        "completion_count=%s output_len_min=%s output_len_max=%s",
                        task_id,
                        task_type,
                        task_level,
                        len(completion_texts),
                        min(output_lengths) if output_lengths else 0,
                        max(output_lengths) if output_lengths else 0,
                    )

                # Compute rewards for all completions in batch
                # Returns combined_scores (includes LLM Judge if batch complete)
                batch_scores = curriculum.compute_rewards(task_id, completion_texts)

                # Map scores back to original indices
                for idx, score in zip(indices, batch_scores):
                    reward_with_index[idx] = float(score)

                if progress_state is not None:
                    progress_state.update(
                        phase="reward_task",
                        event="reward_task_end",
                        curriculum_step=curriculum.global_step,
                        last_reward_task=task_id,
                    )
                if log_reward_batches:
                    logger.info(
                        "reward_task_end task_id=%s duration_sec=%.2f "
                        "curriculum_step=%s scores=%s",
                        task_id,
                        time.monotonic() - task_started,
                        curriculum.global_step,
                        _summarize_scores(batch_scores),
                    )

            except Exception as e:
                logger.exception(
                    "reward_task_failed task_id=%s duration_sec=%.2f",
                    task_id,
                    time.monotonic() - task_started,
                )
                print(f"Warning: Error computing reward for task {task_id}: {e}")
                # Add zero scores for all completions in this task
                for idx, _ in index_completion_list:
                    reward_with_index[idx] = 0.0

        # Reconstruct rewards in original order
        rewards_list = [reward_with_index.get(i, 0.0) for i in range(len(completions))]
        if progress_state is not None:
            progress_state.update(
                phase="reward_batch",
                event="reward_batch_end",
                curriculum_step=curriculum.global_step,
            )
        if log_reward_batches:
            logger.info(
                "reward_batch_end duration_sec=%.2f curriculum_step=%s rewards=%s",
                time.monotonic() - reward_batch_started,
                curriculum.global_step,
                _summarize_scores(rewards_list),
            )
        return rewards_list

    return reward_func


class CurriculumLoggingCallback(TrainerCallback):
    """Callback to log curriculum statistics during training.

    No dataset refresh needed - the dynamic dataset automatically reflects
    the current curriculum level.
    """

    def __init__(self, curriculum: CurriculumLearning, log_every_n_steps: int = 10):
        """Initialize the callback.

        Args:
            curriculum: The CurriculumLearning instance to monitor
            log_every_n_steps: Log curriculum stats every N steps
        """
        self.curriculum = curriculum
        self.log_every_n_steps = log_every_n_steps
        self.last_log_step = 0

    def on_step_end(self, args, state, control, **kwargs):
        """Log curriculum statistics periodically."""
        if state.global_step - self.last_log_step >= self.log_every_n_steps:
            stats = self.curriculum.get_learning_stats()
            current_level = stats.get("current_level", 0)
            judge_stats = (
                self.curriculum.get_judge_scores()
                if self.curriculum.use_llm_judge and wandb.run is not None
                else {"enabled": self.curriculum.use_llm_judge}
            )
            logger.info(
                "curriculum_stats trainer_step=%s curriculum_step=%s current_level=%s "
                "sliding_window=%s judge=%s",
                state.global_step,
                self.curriculum.global_step,
                current_level,
                make_json_serializable(stats.get("sliding_window_stats", {})),
                make_json_serializable(judge_stats),
            )

            # Log to wandb
            if wandb.run is not None:
                wandb_logs = {
                    "curriculum/current_level": current_level,
                    "curriculum/global_step": self.curriculum.global_step,
                }

                # Log success rates by level
                sliding_window = stats.get("sliding_window_stats", {})

                # Log aggregated stats (these are always at top level)
                if "mean_success_rate" in sliding_window:
                    wandb_logs["curriculum/mean_success_rate"] = sliding_window.get(
                        "mean_success_rate", 0
                    )
                    wandb_logs["curriculum/mean_variance"] = sliding_window.get(
                        "mean_variance", 0
                    )
                    wandb_logs["curriculum/total_samples"] = sliding_window.get(
                        "samples", 0
                    )

                # Log per-level stats
                for level_stat in sliding_window.get("by_level", []):
                    level = level_stat.get("level", "unknown")
                    success_rate = level_stat.get("success_rate", 0)
                    variance = level_stat.get("variance", 0)
                    samples = level_stat.get("samples", 0)

                    wandb_logs[f"curriculum/level_{level}/success_rate"] = success_rate
                    wandb_logs[f"curriculum/level_{level}/variance"] = variance
                    wandb_logs[f"curriculum/level_{level}/samples"] = samples

                # Log judge scores (if LLM Judge enabled)
                if self.curriculum.use_llm_judge:
                    wandb_logs["curriculum/judge/avg_judge_score"] = judge_stats.get(
                        "avg_judge_score", 0.0
                    )
                    wandb_logs["curriculum/judge/min_judge_score"] = judge_stats.get(
                        "min_judge_score", 0.0
                    )
                    wandb_logs["curriculum/judge/max_judge_score"] = judge_stats.get(
                        "max_judge_score", 0.0
                    )
                    wandb_logs["curriculum/judge/count"] = judge_stats.get(
                        "judge_score_count", 0
                    )

                # Log generation statistics for recent tasks (Phase 4: Enhanced generation tracking)
                recent_tasks = list(self.curriculum.session.tasks.keys())[
                    -10:
                ]  # Last 10 tasks
                total_generations = 0
                total_correct_generations = 0
                generation_scores = []

                for task_id in recent_tasks:
                    batch_stats = self.curriculum.session.get_batch_stats(task_id)
                    if batch_stats:
                        total_generations += batch_stats["num_generations"]
                        total_correct_generations += batch_stats["correct_generations"]
                        generation_scores.extend(
                            [batch_stats["scores"]["avg"]]
                            if batch_stats["num_generations"] > 0
                            else []
                        )

                        # Log per-task generation stats (sample a few)
                        if (
                            len(recent_tasks) <= 3
                        ):  # Only log detailed stats for very recent tasks
                            wandb_logs[
                                f"generations/task_{task_id}/num_generations"
                            ] = batch_stats["num_generations"]
                            wandb_logs[f"generations/task_{task_id}/avg_score"] = (
                                batch_stats["scores"]["avg"]
                            )
                            wandb_logs[f"generations/task_{task_id}/best_score"] = (
                                batch_stats["scores"]["max"]
                            )
                            wandb_logs[f"generations/task_{task_id}/correct_ratio"] = (
                                batch_stats["correct_generations"]
                                / batch_stats["num_generations"]
                                if batch_stats["num_generations"] > 0
                                else 0
                            )

                # Log aggregated generation statistics
                if total_generations > 0:
                    wandb_logs["generations/recent_tasks_count"] = len(recent_tasks)
                    wandb_logs["generations/total_generations"] = total_generations
                    wandb_logs["generations/avg_correct_ratio"] = (
                        total_correct_generations / total_generations
                    )
                    if generation_scores:
                        wandb_logs["generations/avg_task_score"] = sum(
                            generation_scores
                        ) / len(generation_scores)

                wandb.log(wandb_logs)

            self.last_log_step = state.global_step


class TrainingDiagnosticsCallback(TrainerCallback):
    """File-backed trainer progress logs for detecting hangs."""

    def __init__(
        self,
        progress_state: TrainingProgressState,
        curriculum: CurriculumLearning,
        log_every_n_steps: int = 1,
    ):
        self.progress_state = progress_state
        self.curriculum = curriculum
        self.log_every_n_steps = max(1, log_every_n_steps)

    def _should_log(self, step: int) -> bool:
        return step <= 1 or step % self.log_every_n_steps == 0

    def on_train_begin(self, args, state, control, **kwargs):
        self.progress_state.update(
            phase="training",
            event="train_begin",
            trainer_step=state.global_step,
            curriculum_step=self.curriculum.global_step,
        )
        logger.info(
            "train_begin max_steps=%s num_train_epochs=%s train_batch_size=%s "
            "gradient_accumulation_steps=%s %s",
            state.max_steps,
            args.num_train_epochs,
            args.per_device_train_batch_size,
            args.gradient_accumulation_steps,
            _runtime_diagnostics(),
        )

    def on_step_begin(self, args, state, control, **kwargs):
        next_step = state.global_step + 1
        self.progress_state.update(
            phase="training_step",
            event="step_begin",
            trainer_step=next_step,
            curriculum_step=self.curriculum.global_step,
        )
        if self._should_log(next_step):
            logger.info(
                "step_begin step=%s epoch=%s curriculum_step=%s %s",
                next_step,
                state.epoch,
                self.curriculum.global_step,
                _runtime_diagnostics(),
            )

    def on_step_end(self, args, state, control, **kwargs):
        self.progress_state.update(
            phase="training_step",
            event="step_end",
            trainer_step=state.global_step,
            curriculum_step=self.curriculum.global_step,
        )
        if self._should_log(state.global_step):
            logger.info(
                "step_end step=%s epoch=%s curriculum_step=%s should_save=%s "
                "should_log=%s should_evaluate=%s %s",
                state.global_step,
                state.epoch,
                self.curriculum.global_step,
                control.should_save,
                control.should_log,
                control.should_evaluate,
                _runtime_diagnostics(),
            )

    def on_log(self, args, state, control, logs=None, **kwargs):
        self.progress_state.update(
            phase="training",
            event="trainer_log",
            trainer_step=state.global_step,
            curriculum_step=self.curriculum.global_step,
        )
        logger.info(
            "trainer_log step=%s logs=%s",
            state.global_step,
            json.dumps(make_json_serializable(logs or {}), sort_keys=True),
        )

    def on_save(self, args, state, control, **kwargs):
        self.progress_state.update(
            phase="checkpoint",
            event="save_begin_or_end",
            trainer_step=state.global_step,
            curriculum_step=self.curriculum.global_step,
        )
        logger.info(
            "checkpoint_event step=%s output_dir=%s curriculum_step=%s",
            state.global_step,
            args.output_dir,
            self.curriculum.global_step,
        )

    def on_epoch_begin(self, args, state, control, **kwargs):
        logger.info("epoch_begin epoch=%s step=%s", state.epoch, state.global_step)

    def on_epoch_end(self, args, state, control, **kwargs):
        logger.info("epoch_end epoch=%s step=%s", state.epoch, state.global_step)

    def on_train_end(self, args, state, control, **kwargs):
        self.progress_state.update(
            phase="training",
            event="train_end",
            trainer_step=state.global_step,
            curriculum_step=self.curriculum.global_step,
        )
        logger.info(
            "train_end step=%s curriculum_step=%s %s",
            state.global_step,
            self.curriculum.global_step,
            _runtime_diagnostics(),
        )


def setup_training_args(
    output_dir: str = "./grpo-qwen3-4b",
    learning_rate: float = 1e-5,
    num_train_epochs: int = 3,
    per_device_train_batch_size: int = 4,
    num_generations: int = 4,
    max_completion_length: int = 512,
    max_prompt_length: int = 2048,
    gradient_accumulation_steps: int = 1,
    warmup_steps: int = 0,
    beta: float = 0.1,
    vllm_mode: str = "colocate",
    vllm_server_base_url: Optional[str] = None,
    **kwargs,
) -> GRPOConfig:
    """Setup GRPO training configuration for vLLM.

    Args:
        output_dir: Output directory for checkpoints
        learning_rate: Learning rate for training
        num_train_epochs: Number of training epochs
        per_device_train_batch_size: Batch size per device
        num_generations: Number of completions per prompt
        max_completion_length: Maximum completion length
        max_prompt_length: Maximum prompt length (for vLLM context size)
        gradient_accumulation_steps: Number of steps to accumulate gradients
        warmup_steps: Number of warmup steps for learning rate scheduling
        vllm_mode: vLLM mode ('server' or 'colocate')
        vllm_server_base_url: Base URL for vLLM server mode
        **kwargs: Additional arguments to pass to GRPOConfig

    Returns:
        Configured GRPOConfig instance
    """
    config = GRPOConfig(
        output_dir=output_dir,
        learning_rate=learning_rate,
        num_train_epochs=num_train_epochs,
        per_device_train_batch_size=per_device_train_batch_size,
        num_generations=num_generations,
        max_completion_length=max_completion_length,
        # max_prompt_length=max_prompt_length, # TRL GRPOConfig not supports max_prompt_length
        gradient_accumulation_steps=gradient_accumulation_steps,
        warmup_steps=warmup_steps,
        temperature=0.9,
        top_p=0.9,
        beta=beta,
        repetition_penalty=1.0,  # CRITICAL: Must be 1.0 to match base policy logprobs
        # vLLM configuration
        use_vllm=True,
        vllm_gpu_memory_utilization=0.5,
        vllm_model_impl="transformers",
        vllm_mode=vllm_mode,
        optim="paged_adamw_8bit",
        gradient_checkpointing=True,
        logging_steps=10,
        save_steps=200,
        save_strategy="steps",
        eval_strategy="no",
        log_completions=False,  # avoid Rich library bugs
        num_completions_to_print=0,
        model_init_kwargs={
            "torch_dtype": torch.bfloat16,
            "attn_implementation": "flash_attention_2",
        },
        max_grad_norm=0.5,
        # Below enables GSPO:
        importance_sampling_level="sequence",
        mask_truncated_completions=False,
        vllm_importance_sampling_correction=True,
        **kwargs,
    )

    # Set vLLM server URL if in server mode
    if vllm_mode == "server" and vllm_server_base_url:
        config.vllm_server_base_url = vllm_server_base_url

    return config


def create_lora_config(
    lora_r: int = 32,
    lora_alpha: int = 64,
    lora_dropout: float = 0.05,
    target_modules: Optional[List[str]] = None,
) -> LoraConfig:
    """Create a LoRA configuration for parameter-efficient fine-tuning.

    Args:
        lora_r: LoRA rank (intrinsic dimension)
        lora_alpha: LoRA scaling factor
        lora_dropout: Dropout probability for LoRA layers
        target_modules: List of module names to apply LoRA to

    Returns:
        LoraConfig instance
    """
    if target_modules is None:
        # Default target modules (all attention and MLP projections)
        target_modules = [
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ]

    lora_config = LoraConfig(
        r=lora_r,
        lora_alpha=lora_alpha,
        target_modules=target_modules,
        lora_dropout=lora_dropout,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
        modules_to_save=["embed_tokens", "lm_head"],
    )
    return lora_config


def main():
    parser = argparse.ArgumentParser(
        description="Train Qwen-3-4B with GRPO + vLLM colocate mode + Infinite-RL curriculum. "
        "NOTE: Requires vocabulary pre-expansion using train/model_expand.py"
    )

    # Model arguments
    parser.add_argument(
        "--model_name",
        type=str,
        default="jed351/Qwen3-4B-ChatVector_SFT-from-IT_and_IT",
        help="Model name or path (should be pre-expanded with special tokens)",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./grpo-qwen3-4b",
        help="Output directory for checkpoints",
    )

    # Training arguments
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=1e-5,
        help="Learning rate",
    )
    parser.add_argument(
        "--num_train_epochs",
        type=int,
        default=3,
        help="Number of training epochs",
    )
    parser.add_argument(
        "--per_device_train_batch_size",
        type=int,
        default=4,
        help="Batch size per device",
    )
    parser.add_argument(
        "--num_generations",
        type=int,
        default=4,
        help="Number of completions per prompt. Use 8 for pre-reasoning GRPO.",
    )
    parser.add_argument(
        "--pre_reasoning_dataset",
        type=str,
        default=None,
        help="Hugging Face dataset name or local JSON/JSONL path containing chat-message SFT examples.",
    )
    parser.add_argument(
        "--pre_reasoning_split",
        type=str,
        default="train",
        help="Dataset split when --pre_reasoning_dataset is a Hugging Face dataset name.",
    )
    parser.add_argument(
        "--pre_reasoning_learning_rate",
        type=float,
        default=None,
        help="Probability of sampling pre_reasoning tasks. Defaults to 1.0 when a pre_reasoning dataset is provided, otherwise 0.0.",
    )
    parser.add_argument(
        "--max_completion_length",
        type=int,
        default=768,
        help="Maximum completion length",
    )
    parser.add_argument(
        "--max_prompt_length",
        type=int,
        default=2048,
        help="Maximum prompt length (for vLLM context sizing - limits vLLM KV cache allocation)",
    )
    parser.add_argument(
        "--beta",
        type=float,
        default=0.1,
        help="KL penalty coefficient for GRPO. Use 0.1-0.2 to slow empty-think drift.",
    )
    parser.add_argument(
        "--num_train_samples",
        type=int,
        default=None,
        help="Number of training samples to generate per refresh. "
        "If None, defaults to batch_size * gradient_accumulation_steps * 4 (enough for a few batches)",
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of steps to accumulate gradients before updating weights",
    )

    # vLLM mode arguments
    parser.add_argument(
        "--vllm_mode",
        type=str,
        choices=["server", "colocate"],
        default="colocate",
        help="vLLM mode: 'server' for external vLLM server, 'colocate' for in-process vLLM",
    )
    parser.add_argument(
        "--vllm_server_base_url",
        type=str,
        default=None,
        help="Base URL for vLLM server mode (required when --vllm-mode=server)",
    )

    # Curriculum arguments
    parser.add_argument(
        "--warmup_step",
        type=int,
        default=32,
        help="Number of warmup steps (level 0 only)",
    )
    parser.add_argument(
        "--window_size",
        type=int,
        default=50,
        help="Sliding window size for success rate tracking",
    )
    parser.add_argument(
        "--success_rate_threshold",
        type=float,
        default=0.7,
        help="Success rate threshold for advancing difficulty",
    )
    parser.add_argument(
        "--variance_threshold",
        type=float,
        default=0.15,
        help="Variance threshold for stability check",
    )
    parser.add_argument(
        "--demote_threshold",
        type=float,
        default=0.4,
        help="Success rate threshold for demoting difficulty (demote if success_rate < demote_threshold)",
    )
    parser.add_argument(
        "--reasoning_language",
        type=str,
        default="en",
        choices=["en", "yue", "zh"],
        help="Language for reasoning (CoT) in <think> tags (default: en). Options: en=English, yue=Cantonese, zh=Mandarin",
    )
    parser.add_argument(
        "--reasoning_template",
        action="store_true",
        help="Model uses a reasoning chat template that injects <think> automatically. "
        "When enabled, skip checking for the opening <think> tag (closing </think> is still required).",
    )
    parser.add_argument(
        "--system_prompt",
        action="store_true",
        default=True,
        help="Inject a system prompt instructing the model to reason in the target language. "
        "Enabled by default when reasoning_language != en. Use --no_system_prompt to disable.",
    )
    parser.add_argument(
        "--no_system_prompt",
        action="store_true",
        help="Disable the reasoning language system prompt.",
    )
    parser.add_argument(
        "--log_curriculum_steps",
        type=int,
        default=10,
        help="Log curriculum stats to W&B every N training steps (default: 10)",
    )
    parser.add_argument(
        "--train_log_file",
        type=str,
        default=None,
        help="Detailed training log file. Relative paths are written under output_dir. "
        "Defaults to output_dir/train[.rankN].log.",
    )
    parser.add_argument(
        "--log_level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="Python logging level for the detailed training log (default: INFO).",
    )
    parser.add_argument(
        "--log_train_steps",
        type=int,
        default=1,
        help="Write trainer step begin/end diagnostics every N steps (default: 1).",
    )
    parser.add_argument(
        "--log_heartbeat_seconds",
        type=int,
        default=120,
        help="Write an alive/progress heartbeat every N seconds. Set 0 to disable.",
    )
    parser.add_argument(
        "--disable_reward_batch_logging",
        action="store_true",
        help="Disable detailed reward batch start/end timing logs.",
    )

    # LLM Judge arguments
    parser.add_argument(
        "--use_llm_judge",
        action="store_true",
        help="Enable LLM Judge for auxiliary reward scoring (requires running sglang server)",
    )
    parser.add_argument(
        "--llm_judge_host",
        type=str,
        default="localhost",
        help="Host address of the sglang LLM Judge server (default: localhost)",
    )
    parser.add_argument(
        "--llm_judge_port",
        type=int,
        default=8000,
        help="Port of the sglang LLM Judge server (default: 8000)",
    )
    parser.add_argument(
        "--llm_judge_model",
        type=str,
        default="Skywork/Skywork-Reward-V2-Qwen3-4B",
        help="Model name for LLM Judge (default: Skywork/Skywork-Reward-V2-Qwen3-4B)",
    )

    # LoRA/PEFT arguments
    parser.add_argument(
        "--use_lora",
        action="store_true",
        help="Use LoRA (Low-Rank Adaptation) for parameter-efficient fine-tuning",
    )
    parser.add_argument(
        "--lora_r",
        type=int,
        default=32,
        help="LoRA rank (intrinsic dimension)",
    )
    parser.add_argument(
        "--lora_alpha",
        type=int,
        default=64,
        help="LoRA scaling factor",
    )
    parser.add_argument(
        "--lora_dropout",
        type=float,
        default=0.05,
        help="Dropout probability for LoRA layers",
    )

    # Quantization arguments
    parser.add_argument(
        "--load_in_4bit",
        action="store_true",
        help="Load model in 4-bit quantization using bitsandbytes",
    )
    parser.add_argument(
        "--load_in_8bit",
        action="store_true",
        help="Load model in 8-bit quantization using bitsandbytes",
    )
    parser.add_argument(
        "--bnb_4bit_compute_dtype",
        type=str,
        default="bf16",
        choices=["fp16", "bf16", "fp32"],
        help="Compute dtype for 4-bit quantization (default: bf16)",
    )
    parser.add_argument(
        "--bnb_4bit_quant_type",
        type=str,
        default="nf4",
        choices=["fp4", "nf4"],
        help="Quantization type for 4-bit (default: nf4)",
    )
    parser.add_argument(
        "--bnb_4bit_use_double_quant",
        action="store_true",
        help="Use double quantization for 4-bit (reduces memory further)",
    )

    # GRPO loss type arguments
    parser.add_argument(
        "--dapo",
        action="store_true",
        help="Use DAPO (Decoupled Advantage Policy Optimization) loss",
    )
    parser.add_argument(
        "--enable_thinking",
        action="store_true",
        help="Enable thinking mode via chat_template_kwargs (e.g. for Qwen3 thinking template)",
    )

    # Training control arguments
    parser.add_argument(
        "--max_steps",
        type=int,
        default=None,
        help="Maximum number of training steps (if set, overrides num_train_epochs). "
        "Dataset size will be calculated to match this.",
    )

    # W&B arguments
    parser.add_argument(
        "--use_wandb",
        action="store_true",
        help="Enable Weights & Biases (W&B) logging",
    )
    parser.add_argument(
        "--wandb_project",
        type=str,
        default="infinite-rl-grpo",
        help="W&B project name",
    )
    parser.add_argument(
        "--wandb_entity",
        type=str,
        default=None,
        help="W&B entity (team/user) name",
    )
    parser.add_argument(
        "--wandb_run_name",
        type=str,
        default=None,
        help="Custom run name for W&B",
    )

    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    train_log_path = setup_training_logging(
        output_dir=output_dir,
        train_log_file=args.train_log_file,
        log_level=args.log_level,
    )
    progress_state = TrainingProgressState()
    progress_state.update(phase="startup", event="args_parsed")
    heartbeat = TrainingHeartbeat(
        progress_state,
        interval_seconds=args.log_heartbeat_seconds,
    )
    heartbeat.start()
    logger.info(
        "Parsed CLI arguments: %s",
        json.dumps(make_json_serializable(vars(args)), sort_keys=True),
    )
    print(f"\nDetailed training log: {train_log_path}")

    # Validate quantization arguments
    if args.load_in_4bit and args.load_in_8bit:
        parser.error(
            "Cannot use both --load_in_4bit and --load_in_8bit. Choose one quantization method."
        )

    # Validate vLLM mode arguments
    if args.vllm_mode == "server" and args.vllm_server_base_url is None:
        parser.error("--vllm_server_base_url is required when --vllm-mode=server")

    if args.pre_reasoning_dataset and not args.use_llm_judge:
        parser.error(
            "--pre_reasoning_dataset requires --use_llm_judge because "
            "pre_reasoning uses LLM Judge as the primary score"
        )

    pre_reasoning_learning_rate = args.pre_reasoning_learning_rate
    if pre_reasoning_learning_rate is None:
        pre_reasoning_learning_rate = 1.0 if args.pre_reasoning_dataset else 0.0

    # Calculate dataset size based on training configuration
    # This ensures we generate exactly the number of samples needed
    effective_batch_size = (
        args.per_device_train_batch_size * args.gradient_accumulation_steps
    )

    if args.max_steps is not None:
        # Use max_steps to calculate total samples needed
        num_train_samples = args.max_steps * effective_batch_size
        logger.info(
            "Calculated dataset size from max_steps. num_train_samples=%s max_steps=%s effective_batch_size=%s",
            num_train_samples,
            args.max_steps,
            effective_batch_size,
        )
        print(f"\nCalculated dataset size: {num_train_samples} samples")
        print(
            f"  (max_steps={args.max_steps} * effective_batch={effective_batch_size})"
        )
    else:
        # Use num_train_epochs to calculate samples per epoch
        # Default to a reasonable number of steps per epoch (e.g., 100)
        steps_per_epoch = 100
        num_train_samples = steps_per_epoch * effective_batch_size
        logger.info(
            "Calculated dataset size from epochs. num_train_samples=%s steps_per_epoch=%s effective_batch_size=%s epochs=%s",
            num_train_samples,
            steps_per_epoch,
            effective_batch_size,
            args.num_train_epochs,
        )
        print(f"\nCalculated dataset size: {num_train_samples} samples per epoch")
        print(
            f"  (steps_per_epoch={steps_per_epoch} * effective_batch={effective_batch_size})"
        )
        print(f"  Training for {args.num_train_epochs} epochs")

    # Initialize W&B if enabled
    if args.use_wandb:
        progress_state.update(phase="wandb_init", event="wandb_init_start")
        logger.info(
            "Initializing W&B. project=%s entity=%s run_name=%s",
            args.wandb_project,
            args.wandb_entity,
            args.wandb_run_name,
        )
        wandb_config = {
            "model": args.model_name,
            "learning_rate": args.learning_rate,
            "num_train_epochs": args.num_train_epochs,
            "batch_size": args.per_device_train_batch_size,
            "num_generations": args.num_generations,
            "pre_reasoning_dataset": args.pre_reasoning_dataset,
            "pre_reasoning_learning_rate": pre_reasoning_learning_rate,
            "max_completion_length": args.max_completion_length,
            "max_prompt_length": args.max_prompt_length,
            "beta": args.beta,
            "use_dapo": args.dapo,
            "use_lora": args.use_lora,
            "use_vllm": True,  # Always enabled for train2.py
            "load_in_4bit": args.load_in_4bit,
            "load_in_8bit": args.load_in_8bit,
            "window_size": args.window_size,
            "success_rate_threshold": args.success_rate_threshold,
        }

        wandb.init(
            project=args.wandb_project,
            entity=args.wandb_entity,
            name=args.wandb_run_name,
            config=wandb_config,
            tags=["grpo", "curriculum", "infinite-rl", "vllm-colocate"],
        )
        progress_state.update(phase="wandb_init", event="wandb_init_end")
        logger.info("W&B initialized")
        print("\n✓ W&B initialized for run tracking")

    # Setup paths
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("Infinite-RL + GRPO Training (vLLM Colocate Mode)")
    print("=" * 80)
    logger.info("Starting Infinite-RL + GRPO training setup")

    # Initialize curriculum
    progress_state.update(phase="curriculum_init", event="curriculum_init_start")
    logger.info("Initializing curriculum")
    print("\n1. Initializing Infinite-RL curriculum...")
    curriculum_config = InfiniteRLConfig(
        warmup_step=args.warmup_step,
        window_size=args.window_size,
        success_rate_threshold=args.success_rate_threshold,
        variance_threshold=args.variance_threshold,
        demote_threshold=args.demote_threshold,
        num_generations=args.num_generations,
        reasoning_language=args.reasoning_language,
        reasoning_template=args.reasoning_template,
        use_system_prompt=not args.no_system_prompt,
        use_llm_judge=args.use_llm_judge,
        llm_judge_host=args.llm_judge_host,
        llm_judge_port=args.llm_judge_port,
        llm_judge_model=args.llm_judge_model,
        pre_reasoning_dataset=args.pre_reasoning_dataset,
        pre_reasoning_split=args.pre_reasoning_split,
        pre_reasoning_learning_rate=pre_reasoning_learning_rate,
        log_file=str(output_dir / "curriculum_learning_log.jsonl"),
    )
    curriculum = create_curriculum(curriculum_config)
    progress_state.update(
        phase="curriculum_init",
        event="curriculum_init_end",
        curriculum_step=curriculum.global_step,
    )
    logger.info(
        "Curriculum initialized. stats=%s config=%s",
        make_json_serializable(curriculum.get_learning_stats()),
        make_json_serializable(asdict(curriculum_config)),
    )
    print(f"   ✓ Curriculum initialized")
    print(f"   - Learning stats: {curriculum.get_learning_stats()}")
    if args.pre_reasoning_dataset:
        print(f"   - Pre-reasoning dataset: {args.pre_reasoning_dataset}")
        print(f"   - Pre-reasoning split: {args.pre_reasoning_split}")
        print(f"   - Pre-reasoning sampling rate: {pre_reasoning_learning_rate}")

    # Load model and tokenizer
    progress_state.update(phase="model_load", event="model_load_start")
    logger.info(
        "Loading model and tokenizer. model_name=%s quantized=%s device_map_pending=True",
        args.model_name,
        args.load_in_4bit or args.load_in_8bit,
    )
    print(f"\n2. Loading model & tokenizer: {args.model_name}...")

    # Prepare quantization config if requested
    quantization_config = None
    if args.load_in_4bit or args.load_in_8bit:
        from transformers import BitsAndBytesConfig

        compute_dtype = torch.bfloat16
        if args.bnb_4bit_compute_dtype == "fp16":
            compute_dtype = torch.float16
        elif args.bnb_4bit_compute_dtype == "fp32":
            compute_dtype = torch.float32

        quantization_config = BitsAndBytesConfig(
            load_in_4bit=args.load_in_4bit,
            load_in_8bit=args.load_in_8bit,
            bnb_4bit_compute_dtype=compute_dtype,
            bnb_4bit_quant_type=args.bnb_4bit_quant_type,
            bnb_4bit_use_double_quant=args.bnb_4bit_use_double_quant,
        )
        print(f"   - Using quantization: {'4-bit' if args.load_in_4bit else '8-bit'}")
        print(f"   - Compute dtype: {args.bnb_4bit_compute_dtype}")
        if args.load_in_4bit:
            print(f"   - 4-bit quant type: {args.bnb_4bit_quant_type}")
            print(f"   - Double quant: {args.bnb_4bit_use_double_quant}")

    # In colocate mode, set device_map to None to avoid memory conflicts
    # For quantization, use "auto" device_map for proper GPU placement
    device_map = None if not (args.load_in_4bit or args.load_in_8bit) else "auto"

    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        torch_dtype=torch.bfloat16,
        device_map=device_map,
        quantization_config=quantization_config,
        attn_implementation="flash_attention_2",
    )
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    progress_state.update(phase="model_load", event="model_load_end")
    logger.info(
        "Model and tokenizer loaded. model_type=%s parameter_count=%s vocab_size=%s tokenizer_size=%s %s",
        model.config.model_type,
        sum(p.numel() for p in model.parameters()),
        model.config.vocab_size,
        len(tokenizer),
        _runtime_diagnostics(),
    )
    print(f"   ✓ Model loaded: {model.config.model_type}")
    print(f"   - Parameters: {sum(p.numel() for p in model.parameters()):,}")
    if args.load_in_4bit or args.load_in_8bit:
        print(f"   - Quantization: {'4-bit' if args.load_in_4bit else '8-bit'}")
    # print(f"   ✓ Tokenizer loaded with vocab size: {len(tokenizer)}")
    print(f"   - Model config vocab_size: {model.config.vocab_size}")

    # # Verify vocab consistency
    if len(tokenizer) != model.config.vocab_size:
        logger.warning(
            "Vocab size mismatch. tokenizer_size=%s model_config_vocab_size=%s",
            len(tokenizer),
            model.config.vocab_size,
        )
        print(f"\n   ⚠️  WARNING: Vocab size mismatch!")
        print(
            f"     Tokenizer: {len(tokenizer)}, Model config: {model.config.vocab_size}"
        )
        print(f"     Run train/model_expand.py to fix this issue.")

    # Apply LoRA if requested
    if args.use_lora:
        progress_state.update(phase="lora_init", event="lora_init_start")
        logger.info(
            "Applying LoRA. r=%s alpha=%s dropout=%s",
            args.lora_r,
            args.lora_alpha,
            args.lora_dropout,
        )
        print(f"\n   Applying LoRA (Parameter-Efficient Fine-Tuning)...")
        lora_config = create_lora_config(
            lora_r=args.lora_r,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
        )
        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()
        progress_state.update(phase="lora_init", event="lora_init_end")
        logger.info("LoRA applied. %s", _runtime_diagnostics())
        print(f"   ✓ LoRA applied successfully")

    # Build dynamic dataset that generates prompts on-demand
    progress_state.update(phase="dataset_init", event="dataset_init_start")
    logger.info("Creating dynamic curriculum dataset. num_train_samples=%s", num_train_samples)
    print(f"\n3. Creating dynamic curriculum dataset...")
    train_dataset = build_dynamic_dataset(
        curriculum,
        num_samples=num_train_samples,
    )
    progress_state.update(phase="dataset_init", event="dataset_init_end")
    logger.info("Dynamic dataset created. len=%s", len(train_dataset))
    print(f"   ✓ Dynamic dataset created (size: {len(train_dataset)} samples)")
    print(f"   - Prompts generated on-demand from current curriculum level")
    print(f"   - Dataset size matches training batch configuration")
    print(f"   - No pre-generation or refresh needed!")

    # Create reward function
    progress_state.update(phase="reward_func_init", event="reward_func_init_start")
    logger.info(
        "Creating reward function. reward_batch_logging=%s",
        not args.disable_reward_batch_logging,
    )
    print("\n4. Creating Infinite-RL-based reward function...")
    reward_func = create_curriculum_reward_func(
        curriculum,
        progress_state=progress_state,
        log_reward_batches=not args.disable_reward_batch_logging,
    )
    progress_state.update(phase="reward_func_init", event="reward_func_init_end")
    logger.info("Reward function created")
    print(f"   ✓ Reward function created")

    # Setup training configuration
    progress_state.update(phase="training_args_init", event="training_args_init_start")
    logger.info("Setting up GRPO training configuration")
    print("\n5. Setting up GRPO training configuration (vLLM colocate mode)...")
    grpo_kwargs = {}
    if args.dapo:
        grpo_kwargs["loss_type"] = "dapo"
    if args.max_steps is not None:
        grpo_kwargs["max_steps"] = args.max_steps

    training_args = setup_training_args(
        output_dir=args.output_dir,
        learning_rate=args.learning_rate,
        num_train_epochs=args.num_train_epochs,
        per_device_train_batch_size=args.per_device_train_batch_size,
        num_generations=args.num_generations,
        max_completion_length=args.max_completion_length,
        max_prompt_length=args.max_prompt_length,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        warmup_steps=args.warmup_step,
        beta=args.beta,
        vllm_mode=args.vllm_mode,
        vllm_server_base_url=args.vllm_server_base_url,
        **grpo_kwargs,
    )
    if args.enable_thinking:
        training_args.chat_template_kwargs = {"enable_thinking": True}

    progress_state.update(phase="training_args_init", event="training_args_init_end")
    logger.info(
        "Training configuration ready. config=%s",
        json.dumps(make_json_serializable(training_args.to_dict()), sort_keys=True),
    )
    print(f"   ✓ Training configuration ready")
    print(f"   - Batch size (per device): {training_args.per_device_train_batch_size}")
    print(f"   - Number of generations: {training_args.num_generations}")
    print(f"   - Max completion length: {training_args.max_completion_length}")
    print(f"   - Max prompt length: {args.max_prompt_length}")
    print(f"   - KL beta: {training_args.beta}")
    print(
        f"   - vLLM max_model_len: {args.max_prompt_length + args.max_completion_length} tokens (overrides model default)"
    )
    print(f"   - vLLM GPU memory utilization: 50%")
    print(f"   - Learning rate: {training_args.learning_rate}")
    print(f"   - Warmup steps: {training_args.warmup_steps}")
    print(f"   - vLLM mode: {args.vllm_mode}")
    if args.vllm_mode == "server":
        print(f"   - vLLM server URL: {args.vllm_server_base_url}")
    if args.max_steps:
        print(f"   - Max steps: {args.max_steps}")
    else:
        print(f"   - Num epochs: {args.num_train_epochs}")
    if args.dapo:
        print(f"   - Loss type: DAPO (Decoupled Advantage Policy Optimization)")

    # Initialize trainer
    progress_state.update(phase="trainer_init", event="trainer_init_start")
    logger.info("Initializing GRPO trainer")
    print("\n6. Initializing GRPO trainer...")

    # Setup callbacks for curriculum logging
    callbacks = []
    diagnostics_callback = TrainingDiagnosticsCallback(
        progress_state=progress_state,
        curriculum=curriculum,
        log_every_n_steps=args.log_train_steps,
    )
    callbacks.append(diagnostics_callback)
    print(
        f"   - Added training diagnostics callback (log every {args.log_train_steps} steps)"
    )
    logger.info(
        "Added training diagnostics callback. log_train_steps=%s",
        args.log_train_steps,
    )

    curriculum_callback = CurriculumLoggingCallback(
        curriculum=curriculum,
        log_every_n_steps=args.log_curriculum_steps,
    )
    callbacks.append(curriculum_callback)
    print(
        f"   - Added curriculum logging callback (log every {args.log_curriculum_steps} steps)"
    )

    trainer = GRPOTrainer(
        model=model,
        reward_funcs=[reward_func],  # Pass as list
        args=training_args,
        train_dataset=train_dataset,
        processing_class=tokenizer,
        callbacks=callbacks if callbacks else None,
    )
    progress_state.update(phase="trainer_init", event="trainer_init_end")
    logger.info("Trainer initialized. %s", _runtime_diagnostics())
    print(f"   ✓ Trainer initialized")

    # Save configuration
    config_dict = {
        "model_name": args.model_name,
        "vllm_mode": args.vllm_mode,
        "vllm_server_base_url": (
            args.vllm_server_base_url if args.vllm_mode == "server" else None
        ),
        "load_in_4bit": args.load_in_4bit,
        "load_in_8bit": args.load_in_8bit,
        "quantization_config": (
            {
                "bnb_4bit_compute_dtype": args.bnb_4bit_compute_dtype,
                "bnb_4bit_quant_type": args.bnb_4bit_quant_type,
                "bnb_4bit_use_double_quant": args.bnb_4bit_use_double_quant,
            }
            if args.load_in_4bit
            else None
        ),
        "curriculum_config": asdict(curriculum_config),
        "training_config": training_args.to_dict(),
        "dataset_size": len(train_dataset),
    }

    # Convert to JSON-serializable format (handles dtype objects)
    config_dict = make_json_serializable(config_dict)

    # Create the output directory if it doesn't exist
    output_dir.mkdir(parents=True, exist_ok=True)

    with open(output_dir / "training_config.json", "w") as f:
        json.dump(config_dict, f, indent=2)
    logger.info("Training configuration saved to %s", output_dir / "training_config.json")
    print(f"   ✓ Configuration saved to {output_dir / 'training_config.json'}")

    # Start training
    progress_state.update(phase="training", event="trainer_train_start")
    logger.info("Starting trainer.train()")
    print("\n7. Starting GRPO training...")
    print("=" * 80)

    try:
        trainer.train()
    except Exception:
        logger.exception(
            "trainer.train() failed at trainer_step=%s curriculum_step=%s",
            progress_state.snapshot()["trainer_step"],
            curriculum.global_step,
        )
        raise
    progress_state.update(
        phase="training",
        event="trainer_train_end",
        curriculum_step=curriculum.global_step,
    )
    logger.info("trainer.train() finished. curriculum_step=%s", curriculum.global_step)
    print("=" * 80)
    progress_state.update(phase="model_save", event="final_model_save_start")
    logger.info("Saving final model to %s", output_dir / "final_model")
    trainer.save_model(str(output_dir / "final_model"))
    progress_state.update(phase="model_save", event="final_model_save_end")
    logger.info("Final model saved to %s", output_dir / "final_model")
    print(f"✓ Final model saved to {output_dir / 'final_model'}")

    # Print final curriculum stats
    final_stats = curriculum.get_learning_stats()
    judge_stats = curriculum.get_judge_scores()
    logger.info(
        "Final curriculum statistics. stats=%s judge_stats=%s",
        make_json_serializable(final_stats),
        make_json_serializable(judge_stats),
    )
    print(f"\nFinal Curriculum Statistics:")
    print(f"  - Current level: {final_stats['current_level']}")
    print(f"  - Sliding window stats: {final_stats['sliding_window_stats']}")
    print(
        f"  - Judge scores: avg={judge_stats['avg_judge_score']:.3f}, count={judge_stats['judge_score_count']}"
    )

    # Log final stats to W&B
    if args.use_wandb and wandb.run is not None:
        final_wandb_logs = {
            "curriculum/final_level": final_stats.get("current_level", 0),
        }

        # Log judge scores
        final_wandb_logs["judge/final_avg_score"] = judge_stats.get(
            "avg_judge_score", 0.0
        )
        final_wandb_logs["judge/final_min_score"] = judge_stats.get(
            "min_judge_score", 0.0
        )
        final_wandb_logs["judge/final_max_score"] = judge_stats.get(
            "max_judge_score", 0.0
        )
        final_wandb_logs["judge/final_score_count"] = judge_stats.get(
            "judge_score_count", 0
        )
        final_wandb_logs["judge/final_total_tasks_with_judge"] = judge_stats.get(
            "total_tasks_with_judge", 0
        )

        # Log final success rates and statistics
        sliding_window = final_stats.get("sliding_window_stats", {})

        # Log aggregated stats (these are always at top level)
        if "mean_success_rate" in sliding_window:
            final_wandb_logs["curriculum/final_mean_success_rate"] = sliding_window.get(
                "mean_success_rate", 0
            )
            final_wandb_logs["curriculum/final_mean_variance"] = sliding_window.get(
                "mean_variance", 0
            )
            final_wandb_logs["curriculum/final_total_samples"] = sliding_window.get(
                "samples", 0
            )

        # Log per-level stats
        for level_stat in sliding_window.get("by_level", []):
            level = level_stat.get("level", "unknown")
            success_rate = level_stat.get("success_rate", 0)
            variance = level_stat.get("variance", 0)
            samples = level_stat.get("samples", 0)

            final_wandb_logs[f"final_stats/level_{level}/success_rate"] = success_rate
            final_wandb_logs[f"final_stats/level_{level}/variance"] = variance
            final_wandb_logs[f"final_stats/level_{level}/samples"] = samples

        wandb.log(final_wandb_logs)
        wandb.finish()
        logger.info("Final stats logged to W&B")
        print(f"\n✓ Final stats logged to W&B")

    # Save final stats
    final_stats_with_judge = {**final_stats, "judge_scores": judge_stats}
    with open(output_dir / "final_stats.json", "w") as f:
        json.dump(final_stats_with_judge, f, indent=2, default=str)
    logger.info("Final stats saved to %s", output_dir / "final_stats.json")
    heartbeat.stop()
    print(f"✓ Final stats saved to {output_dir / 'final_stats.json'}")


if __name__ == "__main__":
    main()
