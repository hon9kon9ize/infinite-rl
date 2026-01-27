"""
Curriculum Learning for Infinite RL.

This module provides curriculum learning functionality that progressively
increases task difficulty based on model performance.
"""

import json
import random
from typing import Dict, List, Optional, Union, Any
from pathlib import Path
from .reward_functions import get_reward_functions
from .puzzles import get_available_puzzles, get_puzzle_data


class CurriculumLearning:
    """
    Curriculum learning system that manages task difficulty progression.

    Tracks model performance and adjusts task difficulty from level 1 (easy) to 5 (hard).
    """

    def __init__(
        self,
        timeout: int = 10,
        answer_tag: str = "answer",
        think_tag: str = "think",
        aux_weight: float = 0.3,
        use_lang_consistency: bool = False,
        use_repetition: bool = False,
        use_format: bool = True,
        use_reasoning_steps: bool = False,
        use_length: bool = False,
        lang_consistency_kwargs: Optional[Dict[str, Any]] = None,
        repetition_kwargs: Optional[Dict[str, Any]] = None,
        format_kwargs: Optional[Dict[str, Any]] = None,
        reasoning_steps_kwargs: Optional[Dict[str, Any]] = None,
        length_kwargs: Optional[Dict[str, Any]] = None,
    ):
        """
        Initialize curriculum learning.

        Args:
            timeout: Timeout for reward function execution
            answer_tag: Tag used to extract answers from model responses
            think_tag: Tag used to extract reasoning from model responses
            aux_weight: Weight for auxiliary rewards in combined score (0-1)
            use_lang_consistency: Enable language consistency auxiliary reward
            use_repetition: Enable repetition penalty auxiliary reward
            use_format: Enable format validation auxiliary reward
            use_reasoning_steps: Enable chain-of-thought reasoning steps bonus
            use_length: Enable response length regularizer
            lang_consistency_kwargs: Keyword arguments for LangConsistencyRewardFunction
            repetition_kwargs: Keyword arguments for RepetitionRewardFunction
            format_kwargs: Keyword arguments for FormatRewardFunction
            reasoning_steps_kwargs: Keyword arguments for ReasoningStepsRewardFunction
            length_kwargs: Keyword arguments for LengthRewardFunction
        """
        self.timeout = timeout
        self.answer_tag = answer_tag
        self.think_tag = think_tag
        self.aux_weight = aux_weight
        self.reward_functions = get_reward_functions(
            timeout=timeout, answer_tag=answer_tag, think_tag=think_tag
        )

        # Auxiliary reward functions configuration
        self.use_lang_consistency = use_lang_consistency
        self.use_repetition = use_repetition
        self.use_format = use_format
        self.use_reasoning_steps = use_reasoning_steps
        self.use_length = use_length
        self.lang_consistency_kwargs = lang_consistency_kwargs or {}
        self.repetition_kwargs = repetition_kwargs or {}
        self.format_kwargs = format_kwargs or {}
        self.reasoning_steps_kwargs = reasoning_steps_kwargs or {}
        self.length_kwargs = length_kwargs or {}

        # Initialize auxiliary reward functions
        self.aux_reward_functions: Dict[str, Any] = {}
        self._initialize_aux_reward_functions()

        # Learning state
        self.current_level = 1  # Start at easiest level
        self.task_counters: Dict[str, int] = {}  # {"math": 1, "puzzle": -1, ...}
        self.failed_tasks: Dict[str, str] = {}  # {"task_name": "model_response", ...}
        self.recent_tasks: List[str] = []  # Recently trained tasks for weighting
        self.max_recent_tasks = 50  # Maximum recent tasks to track

        # Load available tasks
        self._load_available_tasks()

    def _initialize_aux_reward_functions(self):
        """Initialize auxiliary reward functions based on configuration."""
        if self.use_lang_consistency:
            try:
                from .reward_functions import LangConsistencyRewardFunction

                self.aux_reward_functions["lang_consistency"] = (
                    LangConsistencyRewardFunction(
                        "lang_consistency",
                        timeout=self.timeout,
                        answer_tag=self.answer_tag,
                        think_tag=self.think_tag,
                        **self.lang_consistency_kwargs,
                    )
                )
            except Exception as e:
                print(
                    f"Warning: Could not initialize LangConsistencyRewardFunction: {e}"
                )

        if self.use_repetition:
            try:
                from .reward_functions import RepetitionRewardFunction

                self.aux_reward_functions["repetition"] = RepetitionRewardFunction(
                    "repetition",
                    timeout=self.timeout,
                    answer_tag=self.answer_tag,
                    think_tag=self.think_tag,
                    **self.repetition_kwargs,
                )
            except Exception as e:
                print(f"Warning: Could not initialize RepetitionRewardFunction: {e}")

        if self.use_format:
            try:
                from .reward_functions import FormatRewardFunction

                self.aux_reward_functions["format"] = FormatRewardFunction(
                    "format",
                    timeout=self.timeout,
                    answer_tag=self.answer_tag,
                    think_tag=self.think_tag,
                    **self.format_kwargs,
                )
            except Exception as e:
                print(f"Warning: Could not initialize FormatRewardFunction: {e}")

        if self.use_reasoning_steps:
            try:
                from .reward_functions import ReasoningStepsRewardFunction

                self.aux_reward_functions["reasoning_steps"] = (
                    ReasoningStepsRewardFunction(
                        "reasoning_steps",
                        timeout=self.timeout,
                        answer_tag=self.answer_tag,
                        think_tag=self.think_tag,
                        **self.reasoning_steps_kwargs,
                    )
                )
            except Exception as e:
                print(
                    f"Warning: Could not initialize ReasoningStepsRewardFunction: {e}"
                )

        if self.use_length:
            try:
                from .reward_functions import LengthRewardFunction

                self.aux_reward_functions["length"] = LengthRewardFunction(
                    "length",
                    timeout=self.timeout,
                    answer_tag=self.answer_tag,
                    think_tag=self.think_tag,
                    **self.length_kwargs,
                )
            except Exception as e:
                print(f"Warning: Could not initialize LengthRewardFunction: {e}")

    def _load_available_tasks(self):
        """Load all available tasks and their ratings."""
        print("DEBUG: Starting task loading")
        self.tasks_by_level: Dict[int, List[Dict[str, Any]]] = {
            i: [] for i in range(1, 6)
        }

        # Load math tasks
        math_file = Path(__file__).parent / "runtimes" / "math.json"
        if math_file.exists():
            try:
                with open(math_file, "r", encoding="utf-8") as f:
                    math_data = json.load(f)
                    for item in math_data:
                        task_info = {
                            "type": "math",
                            "data": item,
                            "rating": item.get("rating", 1),
                            "id": f"math_{hash(str(item))}",
                        }
                        level = min(task_info["rating"], 5)  # Ensure level <= 5
                        self.tasks_by_level[level].append(task_info)
            except Exception as e:
                print(f"Warning: Could not load math tasks: {e}")

        # Load puzzle tasks directly from JSON
        puzzles_file = Path(__file__).parent / "runtimes" / "puzzles.json"
        print(f"DEBUG: Looking for puzzles at: {puzzles_file}")
        print(f"DEBUG: File exists: {puzzles_file.exists()}")
        if puzzles_file.exists():
            try:
                with open(puzzles_file, "r", encoding="utf-8") as f:
                    puzzles_data = json.load(f)

                print(
                    f"DEBUG: Loaded puzzle data with keys: {list(puzzles_data.keys())}"
                )
                for lang in ["javascript", "python"]:
                    if lang in puzzles_data:
                        puzzles_list = puzzles_data[lang]
                        print(f"DEBUG: {lang} has {len(puzzles_list)} puzzles")
                        puzzle_count = 0
                        for puzzle_name, puzzle_info in puzzles_list.items():
                            print(
                                f"DEBUG: Checking puzzle {puzzle_name}: {type(puzzle_info)}"
                            )
                            if (
                                isinstance(puzzle_info, dict)
                                and "rating" in puzzle_info
                            ):
                                print(
                                    f"DEBUG: Adding {puzzle_name} with rating {puzzle_info.get('rating')}"
                                )
                                task_info = {
                                    "type": "puzzle",
                                    "language": lang,
                                    "puzzle_name": puzzle_name,
                                    "data": puzzle_info,
                                    "rating": puzzle_info.get("rating") or 3,
                                    "id": f"puzzle_{lang}_{puzzle_name}",
                                }
                                level = min(task_info["rating"], 5)
                                self.tasks_by_level[level].append(task_info)
                                puzzle_count += 1
                            else:
                                print(
                                    f"DEBUG: Skipping {puzzle_name} - no rating or not dict"
                                )
            except Exception as e:
                print(f"DEBUG: Error loading puzzles: {e}")
                import traceback

                traceback.print_exc()
        else:
            print(f"DEBUG: Puzzles file not found at {puzzles_file}")

        # Print summary
        total_tasks = sum(len(tasks) for tasks in self.tasks_by_level.values())
        print(
            f"Loaded {total_tasks} tasks across {len(self.tasks_by_level)} difficulty levels"
        )
        for level in range(1, 6):
            print(f"  Level {level}: {len(self.tasks_by_level[level])} tasks")

    def compute_reward(
        self,
        task_type: str,
        model_output: str,
        expected_output: Union[str, dict],
        task_id: Optional[str] = None,
    ) -> float:
        """
        Evaluate model output and update learning state.

        Args:
            task_type: Type of task ("math", "puzzle", "python", "javascript")
            model_output: Raw model response
            expected_output: Expected answer
            task_id: Optional task identifier for tracking

        Returns:
            Reward score combining primary correctness and auxiliary metrics
        """
        # Get appropriate reward function
        if task_type not in self.reward_functions:
            raise ValueError(f"Unknown task type: {task_type}")

        reward_fn = self.reward_functions[task_type]

        # Compute primary reward
        result = reward_fn.compute_reward(model_output, expected_output)
        score = result.score

        # Compute auxiliary rewards and combine them
        is_correct = score >= 0.5  # Threshold for success
        aux_score_dict = self.get_aux_reward_scores(
            model_output, expected_output, is_correct=is_correct
        )
        aux_scores = [(name, score) for name, score in aux_score_dict.items()]

        # Combine scores (primary + auxiliary)
        # For now, use a simple average; can be customized
        if aux_scores:
            aux_avg = sum(s for _, s in aux_scores) / len(aux_scores)
            # Combine primary and auxiliary rewards using configurable weight
            primary_weight = 1.0 - self.aux_weight
            combined_score = primary_weight * score + self.aux_weight * aux_avg
        else:
            combined_score = score

        # Update counters using the combined score
        if task_type not in self.task_counters:
            self.task_counters[task_type] = 0

        if combined_score >= 0.5:  # Threshold for success
            self.task_counters[task_type] += 1
        else:  # Incorrect
            self.task_counters[task_type] -= 1
            # Store failed task for reflective learning
            if task_id:
                self.failed_tasks[task_id] = model_output
        # Update recent tasks
        if task_id:
            self.recent_tasks.append(task_id)
            if len(self.recent_tasks) > self.max_recent_tasks:
                self.recent_tasks.pop(0)

        # Check if we should advance level
        self._update_level()

        return combined_score

    def _update_level(self):
        """Update current difficulty level based on performance."""
        # Simple progression logic: advance if mostly successful
        total_score = sum(self.task_counters.values())
        task_types_count = len([c for c in self.task_counters.values() if c != 0])

        if (
            task_types_count > 0 and total_score > task_types_count * 2
        ):  # More successes than failures
            self.current_level = min(self.current_level + 1, 5)

        # Could regress if too many failures, but for now keep it simple
        # if total_score < -task_types_count * 2:
        #     self.current_level = max(self.current_level - 1, 1)

    def get_prompt(self) -> Optional[Dict[str, Any]]:
        """
        Get a task prompt appropriate for current difficulty level.

        Returns:
            Task information dict with prompt, expected output, etc.
        """
        # Get available tasks for current level
        available_tasks = self.tasks_by_level.get(self.current_level, [])

        if not available_tasks:
            # Fallback to any available tasks
            for level in range(1, 6):
                if self.tasks_by_level[level]:
                    available_tasks = self.tasks_by_level[level]
                    break

        if not available_tasks:
            return None

        # Weight against recent tasks
        weights = []
        for task in available_tasks:
            weight = 1.0
            if task["id"] in self.recent_tasks:
                # Reduce weight for recent tasks
                recency_penalty = self.recent_tasks.count(task["id"]) / len(
                    self.recent_tasks
                )
                weight = max(0.1, 1.0 - recency_penalty)
            weights.append(weight)

        # Select task with weighting
        selected_task = random.choices(available_tasks, weights=weights, k=1)[0]

        # Format response based on task type
        if selected_task["type"] == "math":
            # Math task
            math_data = selected_task["data"]
            return {
                "task_type": "math",
                "task_id": selected_task["id"],
                "prompt": math_data.get("problem", ""),
                "expected_output": math_data.get("solution", ""),
                "level": selected_task["rating"],
                "data": math_data,
            }

        elif selected_task["type"] == "puzzle":
            # Puzzle task
            puzzle_data = selected_task["data"]
            prompt = self._format_puzzle_prompt(puzzle_data, selected_task["language"])

            return {
                "task_type": "puzzle",
                "task_id": selected_task["id"],
                "prompt": prompt,
                "expected_output": {
                    "puzzle": selected_task["puzzle_name"],
                    "inputs": {},  # Would need to generate specific inputs
                    "language": selected_task["language"],
                },
                "level": selected_task["rating"],
                "data": puzzle_data,
            }

        return None

    def _format_puzzle_prompt(self, puzzle_data: Dict[str, Any], language: str) -> str:
        """Format a puzzle prompt for the model."""
        name = puzzle_data.get("name", "")
        docstring = puzzle_data.get("docstring", "")
        sat_func = puzzle_data.get("sat", "")
        sol_func = puzzle_data.get("sol", "")
        output_format = ""

        if self.answer_tag is not None and self.answer_tag.startswith("```"):
            output_format = (
                f"\nProvide your solution in markdown code blocks: {self.answer_tag}."
            )
        elif self.answer_tag is not None:
            output_format = f"\nProvide your solution in <{self.answer_tag}> tags."

        prompt = f"""Solve this programming puzzle:

# {name}

{docstring}

Write a function that satisfies the following condition:

```javascript
{sat_func}
```

Your solution should be a {language} function with this signature:

```javascript
{sol_func}
```
{output_format}"""

        return prompt

    def get_learning_stats(self) -> Dict[str, Any]:
        """Get current learning statistics."""
        return {
            "current_level": self.current_level,
            "task_counters": self.task_counters.copy(),
            "failed_tasks_count": len(self.failed_tasks),
            "recent_tasks_count": len(self.recent_tasks),
            "available_tasks_by_level": {
                level: len(tasks) for level, tasks in self.tasks_by_level.items()
            },
            "aux_reward_functions": list(self.aux_reward_functions.keys()),
        }

    def get_aux_reward_scores(
        self,
        model_output: str,
        expected_output: Union[str, dict],
        is_correct: bool = False,
    ) -> Dict[str, float]:
        """
        Compute all auxiliary reward scores for the given model output.

        Args:
            model_output: Raw model response
            expected_output: Expected answer
            is_correct: Whether the primary task was answered correctly

        Returns:
            Dictionary mapping auxiliary reward function names to scores
        """
        aux_scores = {}
        for aux_name, aux_fn in self.aux_reward_functions.items():
            try:
                aux_result = aux_fn.compute_reward(model_output, is_correct=is_correct)
                aux_scores[aux_name] = aux_result.score
            except Exception as e:
                print(f"Warning: Auxiliary reward '{aux_name}' failed: {e}")
                aux_scores[aux_name] = 0.0
        return aux_scores
