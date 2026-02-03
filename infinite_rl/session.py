"""Session tracking helpers for Infinite RL curriculum."""

import statistics
from typing import Any, Dict, List, Optional

from .reward_functions import RewardFunctionScore
from .task import Task


class Session:
    """Manages a session of curriculum learning tasks and rewards."""

    def __init__(self):
        self.tasks: Dict[str, Task] = {}
        self.task_history: List[str] = []  # task_ids in order of addition

    def add_task(self, task: Task) -> None:
        """Add a task to the session."""
        self.tasks[task.task_id] = task
        self.task_history.append(task.task_id)

    def get_task(self, task_id: str) -> Optional[Task]:
        """Retrieve a task by its ID."""
        return self.tasks.get(task_id)

    def set_reward(
        self,
        task_id: str,
        task_rewards: List[RewardFunctionScore],
        model_output: Optional[str] = None,
        is_correct: Optional[bool] = None,
    ) -> None:
        """Set rewards for a task.

        Args:
            task_id: Task identifier
            task_rewards: List of reward scores
            model_output: Model output text
            is_correct: Whether task was solved correctly. If None, infers from primary score.
        """
        task = self.get_task(task_id)
        if not task:
            raise ValueError(f"Task {task_id} not found in session")

        # Use provided is_correct value, or infer from primary score if not provided
        if is_correct is None:
            is_correct = task_rewards[0].score >= 0.5 if task_rewards else False

        # Add all rewards to the task
        for reward in task_rewards:
            task.add_reward(reward, is_correct=is_correct)

        task.model_output = model_output
        task.is_correct = is_correct

    def get_task_rewards(self, task_id: str) -> List[RewardFunctionScore]:
        """Get all rewards for a specific task."""
        task = self.get_task(task_id)
        if not task:
            raise ValueError(f"Task {task_id} not found in session")
        return task.task_rewards

    def task_weights(self) -> Dict[str, float]:
        """
        Calculate task weights based on recent performance.

        Returns a dictionary mapping task_id to weight (1.0 = equal probability).
        Tasks are weighted against recent tasks to promote diversity.
        """
        weights: Dict[str, float] = {}

        for task_id in self.tasks.keys():
            weight = 1.0
            if task_id in self.task_history:
                # Reduce weight for recent tasks
                recency_penalty = self.task_history.count(task_id) / max(
                    len(self.task_history), 1
                )
                weight = max(0.1, 1.0 - recency_penalty)
            weights[task_id] = weight

        return weights

    def get_stats(self) -> Dict[str, Any]:
        """Get session statistics."""
        return {
            "total_tasks": len(self.tasks),
            "tasks_evaluated": len([t for t in self.tasks.values() if t.task_rewards]),
            "total_evaluations": len(self.task_history),
        }

    def get_batch_data(self, task_id: str) -> Optional[List[Dict[str, Any]]]:
        """Get all generations and their data for a task (GRPO batch).

        Args:
            task_id: Task identifier

        Returns:
            List of generation data dictionaries, or None if task not found.
            Each dict contains: output, rewards, primary_score, is_correct, created_at
        """
        task = self.get_task(task_id)
        if not task:
            return None

        return [
            {
                "output": gen.output,
                "rewards": [
                    {
                        "reward_function_name": r.reward_function_name,
                        "score": r.score,
                        "info": r.info,
                    }
                    for r in gen.rewards
                ],
                "primary_score": gen.primary_score,
                "is_correct": gen.is_correct,
                "created_at": gen.created_at.isoformat(),
            }
            for gen in task.generations
        ]

    def get_batch_stats(self, task_id: str) -> Optional[Dict[str, Any]]:
        """Get statistics about all generations for a task (GRPO batch).

        Args:
            task_id: Task identifier

        Returns:
            Dictionary with batch statistics, or None if task not found.
            Includes: num_generations, scores (min/max/avg/std), best_generation, etc.
        """
        task = self.get_task(task_id)
        if not task or not task.generations:
            return None

        scores = [gen.primary_score for gen in task.generations]

        return {
            "num_generations": len(task.generations),
            "scores": {
                "min": min(scores),
                "max": max(scores),
                "avg": statistics.mean(scores),
                "std": statistics.stdev(scores) if len(scores) > 1 else 0.0,
            },
            "best_generation": {
                "index": scores.index(max(scores)),
                "score": max(scores),
                "output": task.generations[scores.index(max(scores))].output,
            },
            "correct_generations": sum(1 for gen in task.generations if gen.is_correct),
            "first_correct_at": next(
                (i for i, gen in enumerate(task.generations) if gen.is_correct),
                None
            ),
        }
