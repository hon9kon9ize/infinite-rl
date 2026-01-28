"""Session tracking helpers for Infinite RL curriculum."""

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
    ) -> None:
        """Set rewards for a task."""
        task = self.get_task(task_id)
        if not task:
            raise ValueError(f"Task {task_id} not found in session")

        # Determine if task is correct (primary score >= 0.5)
        is_correct = task_rewards[0].score >= 0.5 if task_rewards else False

        # Add all rewards to the task
        for reward in task_rewards:
            task.add_reward(reward, is_correct=is_correct)

        task.model_output = model_output

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
