"""
Training Simulator Emulator Package

Provides tools for simulating curriculum learning progression with synthetic response combinations.

Main Components:
- TrainingSimulator: Core simulator for running training scenarios
- ResponsePattern: Predefined response patterns for quick testing
- AdvancedScenarios: Complex training scenarios
"""

from .training_simulator import TrainingSimulator, RewardSnapshot
from .advanced_scenarios import (
    ResponsePattern,
    AdvancedScenarios,
    run_advanced_scenario,
)

__all__ = [
    "TrainingSimulator",
    "RewardSnapshot",
    "ResponsePattern",
    "AdvancedScenarios",
    "run_advanced_scenario",
]
