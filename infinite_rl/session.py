"""Session tracking helpers for Infinite RL curriculum."""

import statistics
from typing import Any, Dict, List, Optional
import json
from pathlib import Path
import random

from .reward_functions import RewardFunctionScore
from .task import Task
from .prompt_templates import (
    format_math_prompt,
    format_puzzle_prompt,
    format_reflective_math_prompt,
    format_reflective_puzzle_prompt,
    format_truthy_judge_system_prompt,
    format_truthy_user_prompt,
)
from .utils.param_extractor import extract_puzzle_inputs


class Session:
    """Manages a session of curriculum learning tasks and rewards."""

    def __init__(
        self,
        answer_tag: str = "answer",
        think_tag: str = "think",
        puzzle_one_shot: bool = False,
    ):
        self.answer_tag = answer_tag
        self.think_tag = think_tag
        self.puzzle_one_shot = puzzle_one_shot
        self.task_instance_counter: int = 0
        self.tasks: Dict[str, Task] = {}
        self.task_history: List[str] = []  # task_ids in order of addition
        self.tasks_by_level: Dict[int, List[Dict[str, Any]]] = {
            i: [] for i in range(0, 7)  # 0-6 level
        }
        self.truthy_tasks: List[Dict[str, Any]] = []
        self._load_available_tasks()

    def _load_available_tasks(self):
        """Load all available tasks and their ratings."""

        # Helper function to load JSON file from package resources
        def load_runtime_json(filename):
            """Load JSON file from runtime resources, trying multiple methods."""
            try:
                # Method 1: Try importlib.resources (Python 3.9+)
                try:
                    from importlib.resources import files

                    try:
                        # Python 3.9+ API
                        data_text = (
                            files("infinite_rl.runtimes")
                            .joinpath(filename)
                            .read_text(encoding="utf-8")
                        )
                        return json.loads(data_text)
                    except AttributeError:
                        # Python 3.7-3.8 API
                        from importlib.resources import read_text

                        data_text = read_text(
                            "infinite_rl.runtimes", filename, encoding="utf-8"
                        )
                        return json.loads(data_text)
                except Exception as resources_error:
                    pass

                # Method 2: Fallback to Path-based loading
                file_path = Path(__file__).parent / "runtimes" / filename
                if file_path.exists():
                    with open(file_path, "r", encoding="utf-8") as f:
                        data = json.load(f)
                        return data
                else:
                    return None
            except Exception as e:
                print(f"ERROR: Could not load {filename}: {e}")
                import traceback

                traceback.print_exc()
                return None

        # Load math tasks
        math_data = load_runtime_json("math.json")
        if math_data:
            try:
                for idx, item in enumerate(math_data):
                    # Add unique dataset ID to prevent collisions
                    dataset_id = f"math_{idx}"
                    task_info = {
                        "type": "math",
                        "data": item,
                        "rating": item.get("rating", 0),
                        "id": dataset_id,  # Unique dataset ID
                    }
                    level = min(task_info["rating"], 6)  # Ensure level <= 6
                    self.tasks_by_level[level].append(task_info)
            except Exception as e:
                print(f"Warning: Could not process math tasks: {e}")
                import traceback

                traceback.print_exc()

        # Load puzzle tasks
        puzzles_data = load_runtime_json("puzzles.json")
        if puzzles_data:
            try:

                for lang in ["javascript", "python"]:
                    if lang in puzzles_data:
                        puzzles_list = puzzles_data[lang]
                        puzzle_count = 0
                        for puzzle_name, puzzle_info in puzzles_list.items():
                            if (
                                isinstance(puzzle_info, dict)
                                and "rating" in puzzle_info
                            ):
                                task_info = {
                                    "type": "puzzle",
                                    "language": lang,
                                    "puzzle_name": puzzle_name,
                                    "data": puzzle_info,
                                    "rating": puzzle_info.get("rating") or 3,
                                    "id": f"puzzle_{lang}_{puzzle_name}",
                                }
                                level = min(task_info["rating"], 6)
                                self.tasks_by_level[level].append(task_info)
                                puzzle_count += 1
            except Exception as e:
                print(f"Warning: Could not process puzzle tasks: {e}")
                import traceback

                traceback.print_exc()

                traceback.print_exc()

        # Load truthy tasks
        truthy_data = load_runtime_json("truthy.json")
        if truthy_data:
            try:
                if isinstance(truthy_data, list):
                    truthy_list = truthy_data
                else:
                    # If it's a dict, extract the list of items
                    truthy_list = (
                        list(truthy_data.values())
                        if isinstance(truthy_data, dict)
                        else []
                    )

                truthy_count = 0

                for idx, truthy_item in enumerate(truthy_list):
                    if isinstance(truthy_item, dict) and "prompt" in truthy_item:
                        task_info = {
                            "type": "truthy",
                            "data": truthy_item,
                            "rating": None,  # Truthy tasks not limited by rating
                            "id": f"truthy_{idx}",  # Unique dataset ID
                        }
                        # Store truthy tasks separately
                        self.truthy_tasks.append(task_info)
                        truthy_count += 1

                print(f"DEBUG: Added {truthy_count} truthy tasks")
            except Exception as e:
                print(f"Warning: Could not process truthy tasks: {e}")
                import traceback

                traceback.print_exc()

        # Print summary
        math_count = sum(
            1
            for level_tasks in self.tasks_by_level.values()
            for task in level_tasks
            if task["type"] == "math"
        )
        puzzle_count = sum(
            1
            for level_tasks in self.tasks_by_level.values()
            for task in level_tasks
            if task["type"] == "puzzle"
        )
        truthy_count = len(self.truthy_tasks)
        total_unique = math_count + puzzle_count + truthy_count
        print(
            f"Loaded {total_unique} unique tasks ({math_count} math, {puzzle_count} puzzles, {truthy_count} truthy)"
        )
        for level in range(0, 7):
            level_tasks = self.tasks_by_level[level]
            math_in_level = sum(1 for t in level_tasks if t["type"] == "math")
            puzzle_in_level = sum(1 for t in level_tasks if t["type"] == "puzzle")
            print(
                f"  Level {level}: {len(level_tasks)} tasks ({math_in_level} math, {puzzle_in_level} puzzles)"
            )

    def create_truthy_task(self, selected_task: Dict[str, Any]) -> Optional[Task]:
        """
        Create a truthy task from selected task data.

        Args:
            selected_task: Task dict with type='truthy', data, and id

        Returns:
            Task object for truthy evaluation, or None on error
        """
        try:
            truthy_data = selected_task["data"]
            base_task_id = selected_task["id"]
            # Generate unique task_id for this instance
            unique_task_id = f"{base_task_id}_{self.task_instance_counter}"
            self.task_instance_counter += 1

            task_name = f"truthy_{truthy_data.get('id', '')}"
            system_prompt = truthy_data.get("system", "")
            user_prompt = truthy_data.get("prompt", "")
            chosen = truthy_data.get("chosen", "")
            rejected = truthy_data.get("rejected", "")

            # Validate critical fields are present and non-empty
            # prompt, chosen, and rejected are required for truthy tasks
            if not user_prompt or not chosen or not rejected:
                raise ValueError("Missing required fields: prompt, chosen, or rejected")

            user_prompt = format_truthy_user_prompt(
                system_prompt, user_prompt, self.think_tag
            )

            # Format the prompt with chosen and rejected options
            judge_system_prompt = format_truthy_judge_system_prompt(
                user_prompt, chosen, rejected
            )

            # Store chosen and rejected in expected_answer for reproducibility
            expected_answer = {
                "chosen": chosen,
                "rejected": rejected,
            }

            task_obj = Task(
                task_id=unique_task_id,
                task_name=task_name,
                task_type="truthy",
                level=-1,  # Random level since truthy not limited by rating
                prompt=user_prompt,
                judge_system_prompt=judge_system_prompt,
                expected_answer=expected_answer,
                reasoning_language=truthy_data.get("reasoning_language", "en"),
                language=truthy_data.get("language", "en"),
                dataset_id=base_task_id,
            )
            self.add_task(task_obj)
            return task_obj
        except Exception as e:
            print(f"Error creating truthy task: {e}")
            return None

    def create_math_task(self, selected_task: Dict[str, Any]) -> Optional[Task]:
        """
        Create a math task from selected task data.

        Args:
            selected_task: Task dict with type='math', data, rating, and id

        Returns:
            Task object for math problem, or None on error
        """
        try:
            math_data = selected_task["data"]
            base_task_id = selected_task["id"]
            # Generate unique task_id for this instance
            unique_task_id = f"{base_task_id}_{self.task_instance_counter}"
            self.task_instance_counter += 1

            problem_statement = math_data.get("prompt", "")
            language = math_data.get("lang", "en")
            prompt = format_math_prompt(
                problem_statement, self.answer_tag, language, self.think_tag
            )
            task_name = f"math_{hash(prompt)}"
            expected_output = math_data.get("response", "")

            task_obj = Task(
                task_id=unique_task_id,
                task_name=task_name,
                task_type="math",
                level=selected_task["rating"],
                prompt=prompt,
                expected_answer=expected_output,
                language=language,
                dataset_id=base_task_id,
            )
            self.add_task(task_obj)
            return task_obj
        except Exception as e:
            print(f"Error creating math task: {e}")
            return None

    def create_puzzle_task(self, selected_task: Dict[str, Any]) -> Optional[Task]:
        """
        Create a puzzle task from selected task data.

        Args:
            selected_task: Task dict with type='puzzle', data, language, puzzle_name, rating, and id

        Returns:
            Task object for puzzle, or None on error
        """
        try:
            puzzle_data = selected_task["data"]
            base_task_id = selected_task["id"]
            # Generate unique task_id for this instance
            unique_task_id = f"{base_task_id}_{self.task_instance_counter}"
            self.task_instance_counter += 1

            task_name = selected_task["puzzle_name"]
            language = selected_task["language"]  # javascript or python
            prompt = format_puzzle_prompt(
                puzzle_data,
                language,
                self.answer_tag,
                self.think_tag,
                self.puzzle_one_shot,
            )
            puzzle_inputs = extract_puzzle_inputs(puzzle_data, language)
            expected_output = {
                "puzzle": selected_task["puzzle_name"],
                "inputs": puzzle_inputs,
                "language": language,
            }

            task_obj = Task(
                task_id=unique_task_id,
                task_name=task_name,
                task_type="puzzle",
                level=selected_task["rating"],
                prompt=prompt,
                expected_answer=expected_output,
                language=language,
                dataset_id=base_task_id,
            )
            self.add_task(task_obj)
            return task_obj
        except Exception as e:
            print(f"Error creating puzzle task: {e}")
            return None

    def _get_recent_task_ids(self) -> List[str]:
        """Get recent task base IDs from session history."""
        return [
            tid.rsplit("_", 1)[0] if "_" in tid else tid for tid in self.task_history
        ]

    def add_task(self, task: Task) -> None:
        """Add a task to the session."""
        self.tasks[task.task_id] = task
        self.task_history.append(task.task_id)

    def get_task(self, task_id: str) -> Optional[Task]:
        """Retrieve a task by its ID."""
        return self.tasks.get(task_id)

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
            "tasks_evaluated": len([t for t in self.tasks.values() if t.generations]),
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
                (i for i, gen in enumerate(task.generations) if gen.is_correct), None
            ),
        }
