"""Generation data model for GRPO batches."""

import datetime
from dataclasses import dataclass, field
from typing import List

from .reward_functions import RewardFunctionScore


@dataclass
class Generation:
    """A single generation in a GRPO batch.

    Represents one model output with its rewards and score.
    """

    output: str
    rewards: List[RewardFunctionScore]
    primary_score: float
    combined_score: float = None
    created_at: datetime.datetime = field(default_factory=datetime.datetime.now)

    @property
    def is_correct(self) -> bool:
        """Whether this generation is considered correct."""
        return self.primary_score >= 0.5

    def to_dict(self) -> dict:
        """Convert to dict for logging."""
        return {
            "output": self.output,
            "rewards": [r.to_dict() for r in self.rewards],
            "primary_score": self.primary_score,
            "combined_score": self.combined_score,
            "is_correct": self.is_correct,
            "created_at": self.created_at.isoformat(),
        }
