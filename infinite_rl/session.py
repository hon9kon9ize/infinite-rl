"""Session tracking helpers for Infinite RL curriculum."""

import statistics
from typing import Any, Dict, List, Optional
import json
from pathlib import Path
import random

from .task import Task
from .prompt_templates import (
    format_math_prompt,
    format_pre_reasoning_judge_system_prompt,
    format_pre_reasoning_prompt,
    format_puzzle_prompt,
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
        reasoning_language: str = "en",
        reasoning_template: bool = False,
        pre_reasoning_dataset: Optional[str] = None,
        pre_reasoning_split: str = "train",
    ):
        self.answer_tag = answer_tag
        self.think_tag = think_tag
        self.reasoning_language = reasoning_language
        self.reasoning_template = reasoning_template
        self.pre_reasoning_dataset = pre_reasoning_dataset
        self.pre_reasoning_split = pre_reasoning_split
        self.task_instance_counter: int = 0
        self.tasks: Dict[str, Task] = {}
        self.task_history: List[str] = []  # task_ids in order of addition
        self.tasks_by_level: Dict[int, List[Dict[str, Any]]] = {
            i: [] for i in range(0, 7)  # 0-6 level
        }
        self.pre_reasoning_tasks: List[Dict[str, Any]] = []
        # Backward-compatible alias for older callers/tests. New code should use
        # pre_reasoning_tasks and task_type="pre_reasoning".
        self.truthy_tasks = self.pre_reasoning_tasks
        self._load_available_tasks()

    @property
    def truthy_tasks(self) -> List[Dict[str, Any]]:
        """Deprecated alias for pre_reasoning_tasks."""
        return self.pre_reasoning_tasks

    @truthy_tasks.setter
    def truthy_tasks(self, value: List[Dict[str, Any]]) -> None:
        self.pre_reasoning_tasks = value

    def _load_pre_reasoning_source(self, source: str) -> List[Dict[str, Any]]:
        """Load pre-reasoning rows from a JSON/JSONL file or HF dataset name."""
        if not source:
            return []

        source_path = Path(source).expanduser()
        if source_path.exists():
            if source_path.suffix.lower() == ".jsonl":
                rows = []
                with open(source_path, "r", encoding="utf-8") as f:
                    for line in f:
                        line = line.strip()
                        if line:
                            rows.append(json.loads(line))
                return rows

            with open(source_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            if isinstance(data, list):
                return data
            if isinstance(data, dict):
                for key in ("data", "train", "rows", "examples"):
                    value = data.get(key)
                    if isinstance(value, list):
                        return value
                return list(data.values())
            return []

        try:
            from datasets import load_dataset
        except ImportError as exc:
            raise ImportError(
                "Loading pre_reasoning from Hugging Face requires the "
                "'datasets' package."
            ) from exc

        dataset = load_dataset(source, split=self.pre_reasoning_split)
        return [dict(row) for row in dataset]

    def _message_role(self, message: Dict[str, Any]) -> str:
        role = (
            message.get("role")
            or message.get("from")
            or message.get("speaker")
            or message.get("author")
            or ""
        )
        role = str(role).lower()
        if role in {"human", "user"}:
            return "user"
        if role in {"gpt", "assistant", "bot", "model"}:
            return "assistant"
        if role == "system":
            return "system"
        return role or "user"

    def _message_content(self, message: Dict[str, Any]) -> str:
        value = (
            message.get("content")
            or message.get("value")
            or message.get("text")
            or message.get("message")
            or ""
        )
        return str(value)

    def _normalize_messages(self, messages: Any) -> List[Dict[str, str]]:
        if not isinstance(messages, list):
            return []

        normalized = []
        for message in messages:
            if not isinstance(message, dict):
                continue
            content = self._message_content(message)
            if not content:
                continue
            normalized.append(
                {
                    "role": self._message_role(message),
                    "content": content,
                }
            )
        return normalized

    def _normalize_pre_reasoning_item(
        self,
        item: Dict[str, Any],
        idx: int,
    ) -> Optional[Dict[str, Any]]:
        """Normalize common SFT/chat schemas into prompt messages + reference."""
        if not isinstance(item, dict):
            return None

        messages = (
            self._normalize_messages(item.get("messages"))
            or self._normalize_messages(item.get("conversations"))
            or self._normalize_messages(item.get("conversation"))
        )

        reference_answer = ""
        prompt_payload: Any = None

        if messages:
            assistant_indices = [
                i for i, message in enumerate(messages) if message["role"] == "assistant"
            ]
            if assistant_indices:
                final_assistant_idx = assistant_indices[-1]
                reference_answer = messages[final_assistant_idx]["content"]
                prompt_payload = messages[:final_assistant_idx]
            else:
                prompt_payload = messages

        if prompt_payload is None:
            prompt = item.get("prompt") or item.get("question") or item.get("input")
            if isinstance(prompt, list):
                prompt_payload = self._normalize_messages(prompt)
            elif prompt:
                system_prompt = item.get("system") or item.get("system_prompt")
                if system_prompt:
                    prompt_payload = [
                        {"role": "system", "content": str(system_prompt)},
                        {"role": "user", "content": str(prompt)},
                    ]
                else:
                    prompt_payload = str(prompt)

        if not reference_answer:
            for key in (
                "reference_answer",
                "answer",
                "response",
                "completion",
                "output",
                "chosen",
            ):
                value = item.get(key)
                if isinstance(value, str) and value.strip():
                    reference_answer = value.strip()
                    break

        if prompt_payload is None:
            return None

        language = item.get("lang") or item.get("language") or "en"
        dataset_id = str(
            item.get("id")
            or item.get("task_id")
            or item.get("conversation_id")
            or idx
        )

        return {
            "id": dataset_id,
            "prompt": prompt_payload,
            "reference_answer": reference_answer,
            "language": language,
            "raw": item,
        }

    def _add_pre_reasoning_rows(
        self,
        rows: List[Dict[str, Any]],
        id_prefix: str = "pre_reasoning",
    ) -> int:
        count = 0
        for idx, item in enumerate(rows):
            normalized = self._normalize_pre_reasoning_item(item, idx)
            if not normalized:
                continue

            task_info = {
                "type": "pre_reasoning",
                "data": normalized,
                "rating": None,
                "id": f"{id_prefix}_{idx}",
            }
            self.pre_reasoning_tasks.append(task_info)
            count += 1
        return count

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
                    if item.get("lang", "en") != "en":
                        continue

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

        # Load pre-reasoning tasks. Prefer an explicit JSONL/HF dataset when
        # provided, otherwise fall back to packaged runtime data. The older
        # truthy.json runtime is treated as pre_reasoning data for compatibility.
        try:
            pre_reasoning_count = 0
            if self.pre_reasoning_dataset:
                rows = self._load_pre_reasoning_source(self.pre_reasoning_dataset)
                pre_reasoning_count += self._add_pre_reasoning_rows(rows)
            else:
                pre_reasoning_data = load_runtime_json("pre_reasoning.json")
                if pre_reasoning_data is None:
                    pre_reasoning_data = load_runtime_json("truthy.json")

                if isinstance(pre_reasoning_data, list):
                    pre_reasoning_rows = pre_reasoning_data
                elif isinstance(pre_reasoning_data, dict):
                    pre_reasoning_rows = list(pre_reasoning_data.values())
                else:
                    pre_reasoning_rows = []

                pre_reasoning_count += self._add_pre_reasoning_rows(
                    pre_reasoning_rows
                )

            if pre_reasoning_count:
                print(f"DEBUG: Added {pre_reasoning_count} pre_reasoning tasks")
        except Exception as e:
            print(f"Warning: Could not process pre_reasoning tasks: {e}")
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
        pre_reasoning_count = len(self.pre_reasoning_tasks)
        total_unique = math_count + puzzle_count + pre_reasoning_count
        print(
            f"Loaded {total_unique} unique tasks ({math_count} math, {puzzle_count} puzzles, {pre_reasoning_count} pre_reasoning)"
        )
        for level in range(0, 7):
            level_tasks = self.tasks_by_level[level]
            math_in_level = sum(1 for t in level_tasks if t["type"] == "math")
            puzzle_in_level = sum(1 for t in level_tasks if t["type"] == "puzzle")
            print(
                f"  Level {level}: {len(level_tasks)} tasks ({math_in_level} math, {puzzle_in_level} puzzles)"
            )

    def create_pre_reasoning_task(self, selected_task: Dict[str, Any]) -> Optional[Task]:
        """
        Create a pre-reasoning task from chat/SFT task data.

        Args:
            selected_task: Task dict with type='pre_reasoning', data, and id

        Returns:
            Task object for pre-reasoning training, or None on error
        """
        try:
            task_data = selected_task["data"]
            base_task_id = selected_task["id"]
            unique_task_id = f"{base_task_id}_{self.task_instance_counter}"
            self.task_instance_counter += 1

            prompt_payload = task_data["prompt"]
            reference_answer = task_data.get("reference_answer", "")
            language = task_data.get("language", "en")
            prompt = format_pre_reasoning_prompt(
                prompt_payload,
                self.answer_tag,
                self.think_tag,
                reasoning_language=self.reasoning_language,
                reasoning_template=self.reasoning_template,
            )

            judge_system_prompt = None
            if reference_answer:
                judge_system_prompt = format_pre_reasoning_judge_system_prompt(
                    reference_answer,
                    language,
                )

            expected_answer = {
                "reference_answer": reference_answer,
                "language": language,
                "raw": task_data.get("raw", {}),
            }

            task_obj = Task(
                task_id=unique_task_id,
                task_name=f"pre_reasoning_{task_data.get('id', '')}",
                task_type="pre_reasoning",
                level=-1,
                prompt=prompt,
                judge_system_prompt=judge_system_prompt,
                expected_answer=expected_answer,
                language=language,
                reasoning_language=self.reasoning_language,
                dataset_id=base_task_id,
            )
            self.add_task(task_obj)
            return task_obj
        except Exception as e:
            print(f"Error creating pre_reasoning task: {e}")
            return None

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
            language = truthy_data.get("lang", "en")

            # Validate critical fields are present and non-empty
            # prompt, chosen, and rejected are required for truthy tasks
            if not user_prompt or not chosen or not rejected:
                raise ValueError("Missing required fields: prompt, chosen, or rejected")

            user_prompt = format_truthy_user_prompt(
                system_prompt, user_prompt, self.think_tag, language,
                reasoning_language=self.reasoning_language,
            )

            # Format the prompt with chosen and rejected options
            judge_system_prompt = format_truthy_judge_system_prompt(
                user_prompt, chosen, rejected, language
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
                reasoning_language=self.reasoning_language,
                language=truthy_data.get("lang", "en"),
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
                problem_statement, self.answer_tag, language, self.think_tag,
                reasoning_language=self.reasoning_language,
                reasoning_template=self.reasoning_template,
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
                reasoning_language=self.reasoning_language,
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
                reasoning_language=self.reasoning_language,
                reasoning_template=self.reasoning_template,
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
                reasoning_language=self.reasoning_language,
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
