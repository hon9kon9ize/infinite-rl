"""
Dynamic Curriculum Dataset for GRPO Training.

This module provides a dataset that generates prompts on-demand from the
curriculum learning system, ensuring proper GRPO batching where multiple
completions share the same prompt.
"""

from typing import Any, Dict, List, TYPE_CHECKING
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

    def __getitem__(self, idx):
        """Generate a prompt on-demand from the curriculum.

        For GRPO: Ensures all num_generations completions use the SAME task.
        Example with num_generations=4:
        - idx=0,1,2,3 all get task from batch_idx=0
        - idx=4,5,6,7 all get task from batch_idx=1
        """
        # Calculate which GRPO batch this index belongs to
        batch_idx = idx // self.num_generations

        # Generate task once per batch, reuse for all completions
        if batch_idx not in self.task_cache:
            task = self.curriculum.get_prompt()
            if task is None:
                # Provide detailed debugging info
                stats = self.curriculum.get_learning_stats()
                raise RuntimeError(
                    f"Failed to generate task at batch {batch_idx}. "
                    f"Current level: {stats.get('current_level')}, "
                    f"Available tasks by level: {stats.get('available_tasks_by_level')}, "
                    f"Recent tasks: {stats.get('recent_tasks_count')}, "
                    f"This usually indicates all tasks are marked as recent or all_available_tasks is empty."
                )
            self.task_cache[batch_idx] = task

            # Clean up old caches to prevent memory leak
            # Keep last 60 batches (increased from 50)
            # Only cleanup batches that are far behind current batch_idx
            if len(self.task_cache) > 60:
                # Remove batches that are more than 60 batches behind
                stale_batches = [
                    b for b in self.task_cache.keys() if b < batch_idx - 60
                ]
                for stale_batch in stale_batches:
                    del self.task_cache[stale_batch]

        # Defensive: re-check after cleanup (handles race conditions)
        if batch_idx not in self.task_cache:
            task = self.curriculum.get_prompt()
            if task is None:
                stats = self.curriculum.get_learning_stats()
                raise RuntimeError(
                    f"Failed to generate task at batch {batch_idx} (after cleanup). "
                    f"Current level: {stats.get('current_level')}, "
                    f"Available tasks by level: {stats.get('available_tasks_by_level')}, "
                    f"Recent tasks: {stats.get('recent_tasks_count')}"
                )
            self.task_cache[batch_idx] = task

        task = self.task_cache[batch_idx]

        # Format as TRL expects: list of message dicts
        # Add system prompt with reasoning language instruction when enabled
        # and reasoning language is not English (default)
        messages = []
        if (getattr(self.curriculum, 'use_system_prompt', True)
                and task.reasoning_language
                and task.reasoning_language != "en"
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
