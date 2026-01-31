"""
Dynamic Curriculum Dataset for GRPO Training.

This module provides a dataset that generates prompts on-demand from the
curriculum learning system, ensuring proper GRPO batching where multiple
completions share the same prompt.
"""

from typing import Any, Dict
import json


def get_base_class():
    try:
        import torch.utils.data

        return torch.utils.data.Dataset
    except (ImportError, ModuleNotFoundError):
        # Scenario 2: CI/Environment without PyTorch
        class MockDataset:
            """Minimal shim to satisfy inheritance without Torch."""

            pass

        return MockDataset


class DynamicCurriculumDataset(get_base_class()):
    """Dynamic dataset that generates prompts on-demand from curriculum.

    This ensures each sample reflects the current curriculum level without
    needing to pre-generate or refresh the dataset.

    IMPORTANT: For GRPO training with num_generations > 1, this dataset
    ensures the SAME prompt is reused for all completions in a batch.

    Example with num_generations=4:
        - Indices 0,1,2,3 all get task from batch_idx=0 (same prompt)
        - Indices 4,5,6,7 all get task from batch_idx=1 (same prompt)
        - etc.
    """

    def __init__(self, curriculum: "CurriculumLearning", num_samples: int = 10000):
        """Initialize the dynamic dataset.

        Args:
            curriculum: CurriculumLearning instance to generate prompts from
            num_samples: Virtual size of the dataset (for trainer iteration)
        """
        self.curriculum = curriculum
        self.num_samples = num_samples
        self.num_generations = curriculum.num_generations
        self.task_cache: Dict[int, Any] = {}  # Cache tasks per GRPO batch

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
                raise RuntimeError(f"Failed to generate task at batch {batch_idx}")
            self.task_cache[batch_idx] = task

            # Clean up old caches to prevent memory leak
            # Keep last 50 batches (conservative for multi-worker data loading)
            # Only cleanup batches that are far behind current batch_idx
            if len(self.task_cache) > 50:
                # Remove batches that are more than 30 batches behind
                stale_batches = [
                    b for b in self.task_cache.keys() if b < batch_idx - 30
                ]
                for stale_batch in stale_batches:
                    del self.task_cache[stale_batch]

        # Defensive: re-check after cleanup (handles race conditions)
        if batch_idx not in self.task_cache:
            task = self.curriculum.get_prompt()
            if task is None:
                raise RuntimeError(f"Failed to generate task at batch {batch_idx}")
            self.task_cache[batch_idx] = task

        task = self.task_cache[batch_idx]

        # Format as TRL expects: list of message dicts
        prompt = [{"role": "user", "content": task.prompt}]

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
