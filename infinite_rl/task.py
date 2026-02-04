"""Task metadata helpers for Infinite RL curriculum."""

import datetime
from typing import Any, Dict, List, Optional, Union

from .generation import Generation
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
        judge_system_prompt: Optional[str] = None,
        model_output: Optional[str] = None,
        created_at: Optional[datetime.datetime] = None,
        first_response_at: Optional[datetime.datetime] = None,
        language: Optional[str] = None,
        reasoning_language: Optional[str] = None,
        dataset_id: Optional[str] = None,
    ):
        self.task_id = task_id
        self.task_name = task_name
        self.task_type = task_type
        self.level = level
        self.prompt = prompt
        self.judge_system_prompt = judge_system_prompt
        self.expected_answer = expected_answer
        self.is_correct: Optional[bool] = None  # Track if task was solved correctly
        self.model_output: Optional[str] = model_output
        self.created_at: datetime.datetime = created_at or datetime.datetime.now()
        self.first_response_at: Optional[datetime.datetime] = first_response_at
        self.language: Optional[str] = language
        # For puzzles: language is programming language (javascript/python), reasoning_language is the <think> tag language
        # For math: language is reasoning language, reasoning_language defaults to same
        self.reasoning_language: Optional[str] = reasoning_language or language or "en"
        self.dataset_id = dataset_id

        # NEW: Clean generation hierarchy
        self.generations: List[Generation] = []

    def get_score(self) -> float:
        """Get the primary reward score from the latest generation."""
        latest = self.latest_generation
        return latest.primary_score if latest else 0.0

    def add_generation(
        self, output: str, rewards: List[RewardFunctionScore], primary_score: float
    ) -> Generation:
        """Add a generation to this task."""
        gen = Generation(output=output, rewards=rewards, primary_score=primary_score)
        self.generations.append(gen)
        if self.first_response_at is None:
            self.first_response_at = datetime.datetime.now()
        return gen

    @property
    def latest_generation(self) -> Optional[Generation]:
        """Get the most recent generation."""
        return self.generations[-1] if self.generations else None

    def to_dict(self) -> Dict[str, Any]:
        """Convert task to dictionary for logging.

        Returns task-level metadata only. Generation-level information
        (expected_answer, model_output, is_correct, rewards, scores) is in generations[].
        """
        return {
            "task_id": self.task_id,
            "dataset_id": self.dataset_id,
            "task_name": self.task_name,
            "task_type": self.task_type,
            "level": self.level,
            "language": self.language,
            "reasoning_language": self.reasoning_language,
            "prompt": self.prompt,
            "judge_system_prompt": self.judge_system_prompt,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "first_response_at": (
                self.first_response_at.isoformat() if self.first_response_at else None
            ),
            "generations": [g.to_dict() for g in self.generations],
        }
