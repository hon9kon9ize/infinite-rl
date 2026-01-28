"""Task metadata helpers for Infinite RL curriculum."""

import datetime
from typing import Any, Dict, List, Optional, Union

from .reward_functions import RewardFunctionScore


class Task:
    """Represents a single task with its metadata and reward history."""

    def __init__(
        self,
        task_id: str,
        task_name: str,
        task_type: str,
        level: int,
        prompt: str,
        expected_answer: Union[str, dict],
        task_rewards: Optional[List[RewardFunctionScore]] = None,
        model_output: Optional[str] = None,
        created_at: Optional[datetime.datetime] = None,
        first_response_at: Optional[datetime.datetime] = None,
        language: Optional[str] = None,
    ):
        self.task_id = task_id
        self.task_name = task_name
        self.task_type = task_type
        self.level = level
        self.prompt = prompt
        self.expected_answer = expected_answer
        self.task_rewards: List[RewardFunctionScore] = task_rewards or []
        self.is_correct: Optional[bool] = None  # Track if task was solved correctly
        self.model_output: Optional[str] = model_output
        self.created_at: datetime.datetime = created_at or datetime.datetime.now()
        self.first_response_at: Optional[datetime.datetime] = first_response_at
        self.language: Optional[str] = language

    def add_reward(
        self, task_reward: RewardFunctionScore, is_correct: bool = False
    ) -> None:
        """Add a reward to the task.

        Args:
            task_reward: The reward function score
            is_correct: Whether the task was answered correctly (primary score >= 0.5)
        """
        was_empty = len(self.task_rewards) == 0
        self.task_rewards.append(task_reward)
        if self.is_correct is None:
            self.is_correct = is_correct
        if was_empty and self.first_response_at is None:
            self.first_response_at = datetime.datetime.now()

    def get_score(self) -> float:
        """Get the primary reward score (first reward in the list)."""
        if not self.task_rewards:
            return 0.0
        return self.task_rewards[0].score

    def to_dict(self) -> Dict[str, Any]:
        """Convert task to dictionary for logging."""
        return {
            "task_id": self.task_id,
            "task_name": self.task_name,
            "task_type": self.task_type,
            "level": self.level,
            "language": self.language,
            "prompt": self.prompt,
            "expected_answer": self.expected_answer,
            "model_output": self.model_output,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "first_response_at": (
                self.first_response_at.isoformat() if self.first_response_at else None
            ),
            "is_correct": self.is_correct,
            "task_rewards": [
                {
                    "reward_function_name": r.reward_function_name,
                    "score": r.score,
                    "info": r.info,
                }
                for r in self.task_rewards
            ],
        }
