"""
Dynamic Curriculum Dataset for GRPO Training.

This module provides a dataset that generates prompts on-demand from the
curriculum learning system, ensuring proper GRPO batching where multiple
completions share the same prompt.
"""

from typing import Any, Dict, List, Optional, TYPE_CHECKING
from copy import deepcopy
import json

if TYPE_CHECKING:
    from .curriculum import CurriculumLearning

try:
    import torch.utils.data

    _BaseDataset = torch.utils.data.Dataset
except ImportError:

    class _BaseDataset:
        """Fallback base dataset if PyTorch is not available."""

        def __init__(self):
            pass

        def __len__(self):
            raise NotImplementedError

        def __getitem__(self, idx):
            raise NotImplementedError


class DynamicCurriculumDataset(_BaseDataset):
    """Dynamic dataset that generates prompts on-demand from curriculum.

    This ensures each sample reflects the current curriculum level without
    needing to pre-generate or refresh the dataset.

    IMPORTANT: For GRPO training with num_generations > 1, this dataset
    ensures the SAME prompt is reused for all completions in a batch.

    Example with num_generations=4:
        - Indices 0,1,2,3 all get task from batch_idx=0 (same prompt)
        - Indices 4,5,6,7 all get task from batch_idx=1 (same prompt)
        - etc.

    CRITICAL: num_workers must be 0 to ensure prompt consistency across workers.
    With num_workers > 0, each worker generates its own task independently, breaking
    GRPO's requirement that all generations share the same prompt.
    """

    def __init__(
        self,
        curriculum: "CurriculumLearning",
        num_samples: int = 10000,
        dataloader_num_workers: int = 0,
    ):
        """Initialize the dynamic dataset.

        Args:
            curriculum: CurriculumLearning instance to generate prompts from
            num_samples: Virtual size of the dataset (for trainer iteration)
            dataloader_num_workers: Number of worker processes for DataLoader.
                                   Must be 0 to avoid GRPO prompt consistency issues.
        """
        # FIX #1: Assert num_workers=0 to ensure prompt consistency
        if dataloader_num_workers > 0:
            import warnings
            warnings.warn(
                "DynamicCurriculumDataset is not safe with num_workers > 0 — "
                "each worker generates its own task, breaking GRPO prompt consistency. "
                "Set dataloader_num_workers=0 or pre-generate tasks externally before spawning workers."
            )

        self.curriculum = curriculum
        self.num_samples = num_samples
        self.num_generations = curriculum.num_generations
        self.dataloader_num_workers = dataloader_num_workers
        self.task_cache: Dict[int, Any] = {}  # Cache tasks per GRPO batch

        # Larger eviction window to prevent dropping active batches
        # This is a defensive measure; the root issue is lazy generation in workers
        self.cache_eviction_window = 60  # Keep last 60 batches (was 30)

    def __len__(self):
        return self.num_samples

    def _create_task_synced(self, batch_idx: int) -> None:
        """Deterministic, rank-consistent task selection with no collective ops.

        Seeds Python random with batch_idx so all ranks pick the same puzzle
        without any broadcast. Safe to call from the DataLoader — no collective
        ops means no deadlock when ranks are between FSDP sync points.

        Correctness: _get_recent_task_ids() strips the counter suffix to the
        base task ID, so recent-task tracking stays identical across ranks.
        """
        import random
        global_step = getattr(self.curriculum, "global_step", 0)
        seed = (batch_idx * 7919 + global_step * 104729) & 0xFFFF_FFFF
        saved_state = random.getstate()
        random.seed(seed)
        try:
            task = self.curriculum.get_prompt()
        finally:
            random.setstate(saved_state)

        if task is None:
            stats = self.curriculum.get_learning_stats()
            raise RuntimeError(
                f"Failed to generate task at batch {batch_idx}. "
                f"Level: {stats.get('current_level')}, "
                f"Recent tasks: {stats.get('recent_tasks_count')}"
            )
        self.task_cache[batch_idx] = task
        if len(self.task_cache) > 60:
            for b in [b for b in self.task_cache if b < batch_idx - 60]:
                del self.task_cache[b]

    def __getitem__(self, idx):
        """Generate a prompt on-demand from the curriculum.

        For GRPO: Ensures all num_generations completions use the SAME task.
        Example with num_generations=4:
        - idx=0,1,2,3 all get task from batch_idx=0
        - idx=4,5,6,7 all get task from batch_idx=1

        Multi-GPU safety: uses seed-based determinism so every rank picks the
        same task without any distributed collective (broadcast would deadlock
        when ranks are between FSDP gradient-sync points).
        """
        # Calculate which GRPO batch this index belongs to
        batch_idx = idx // self.num_generations

        # Generate task once per batch, reuse for all completions
        if batch_idx not in self.task_cache:
            self._create_task_synced(batch_idx)

        # Defensive: re-check after cleanup (handles race conditions)
        if batch_idx not in self.task_cache:
            self._create_task_synced(batch_idx)

        task = self.task_cache[batch_idx]

        # Format as TRL expects: list of message dicts
        # Add system prompt with reasoning language instruction when enabled
        # and reasoning language is not English (default)
        if isinstance(task.prompt, list):
            messages = deepcopy(task.prompt)
        else:
            messages = []
            if (getattr(self.curriculum, 'use_system_prompt', True)
                    and task.reasoning_language
                    and task.task_type in ("math", "puzzle")):
                from .prompt_templates import create_reasoning_language_system_prompt
                system_prompt = create_reasoning_language_system_prompt(
                    task.reasoning_language,
                    self.curriculum.think_tag,
                )
                messages.append({"role": "system", "content": system_prompt})

            messages.append({"role": "user", "content": task.prompt})
        prompt = messages

        # Metadata for reward function
        task_metadata = {
            "task_id": task.task_id,
            "task_name": task.task_name,
            "task_type": task.task_type,
            "level": task.level,
            "language": task.language or "python",
            "expected_answer": (
                task.expected_answer
                if isinstance(task.expected_answer, str)
                else json.dumps(task.expected_answer)
            ),
        }

        return {
            "prompt": prompt,
            "task_metadata": task_metadata,
        }
