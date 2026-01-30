"""
Curriculum Learning for Infinite RL.

This module provides curriculum learning functionality that progressively
increases task difficulty based on model performance using a sliding window
success rate with variance tracking.
"""

import json
import random
from collections import deque
from typing import Dict, List, Optional, Union, Any
from pathlib import Path
import datetime
import statistics
from .reward_functions import get_reward_functions, RewardFunctionScore
from .session import Session
from .task import Task
from .utils.param_extractor import extract_puzzle_inputs
from .prompt_templates import (
    format_math_prompt,
    format_puzzle_prompt,
    format_reflective_math_prompt,
    format_reflective_puzzle_prompt,
)


class CurriculumLearning:

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
        log_file: Optional[str] = None,
        window_size: int = 50,
        success_rate_threshold: float = 0.8,
        variance_threshold: float = 0.05,
        demote_threshold: float = 0.4,
        warmup_step: int = 32,
        reflective_learning_rate: float = 0.2,
        level_change_cooldown: int = 5,
        num_generations: int = 4,
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
            log_file: Path to the logging file (JSON Lines format). If None, defaults to 'curriculum_log.jsonl' in the module directory.
            window_size: Size of the sliding window for success rate tracking (default: 50)
            success_rate_threshold: Required success rate for difficulty increase (default: 0.8 = 80%)
            variance_threshold: Maximum variance for success rate stability (default: 0.05)
            demote_threshold: Success rate threshold for difficulty decrease (default: 0.4 = 40%)
            warmup_step: Number of initial steps to only use level 0 tasks (default: 32)
            reflective_learning_rate: Probability of triggering reflective learning on format failures (default: 0.1). Set to 0 to disable.
            level_change_cooldown: Minimum steps between level changes to prevent rapid fluctuations (default: 5)
            num_generations: Number of generations per prompt for GRPO batching (default: 4)
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
        self.current_level = 0  # Start at level 0 (math tasks only)
        self.task_instance_counter: int = 0  # Counter for unique task IDs

        # Warmup stage configuration
        self.warmup_step = warmup_step
        self.global_step: int = 0  # Counter for total steps

        # Reflective learning configuration
        self.reflective_learning_rate = reflective_learning_rate

        # Session management
        self.session = Session()
        self.log_file = Path(log_file) if log_file is not None else None

        # Sliding window tracking for success rate
        self.window_size = window_size
        self.success_rate_threshold = success_rate_threshold
        self.variance_threshold = variance_threshold
        self.demote_threshold = demote_threshold
        # Deques to track success/failure per task type
        self.success_windows: Dict[str, deque] = {}  # Maps task_type -> deque of 0s/1s

        # Level change cooldown: prevent successive advancements too quickly
        self.last_level_change_step: int = -999  # Track when last level change occurred
        self.level_change_cooldown: int = level_change_cooldown

        # GRPO batch tracking: accumulate group responses until prompt-level decision
        self.grpo_batch_scores: Dict[str, List[float]] = (
            {}
        )  # Maps task_id -> list of scores
        self.num_generations: int = (
            num_generations  # Number of generations per prompt (configurable)
        )

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
                        tag_excluded=False,
                        target_tag=self.think_tag,
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

                # Create two format reward functions - one for each tag
                # FormatRewardFunction inherits from RewardFunction which accepts target_tag
                self.aux_reward_functions["format_think"] = FormatRewardFunction(
                    task_name="format_think",
                    timeout=self.timeout,
                    answer_tag=self.answer_tag,
                    think_tag=self.think_tag,
                    target_tag=self.think_tag,
                )

                self.aux_reward_functions["format_answer"] = FormatRewardFunction(
                    task_name="format_answer",
                    timeout=self.timeout,
                    answer_tag=self.answer_tag,
                    think_tag=self.think_tag,
                    target_tag=self.answer_tag,
                )
            except Exception as e:
                print(f"Warning: Could not initialize FormatRewardFunction: {e}")
                import traceback

                traceback.print_exc()

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
        self.tasks_by_level: Dict[int, List[Dict[str, Any]]] = {
            i: [] for i in range(0, 6)  # 0-5 level
        }

        # Load math tasks (downloaded via setup.py into runtimes/)
        math_file = Path(__file__).parent / "runtimes" / "math.json"
        if math_file.exists():
            try:
                with open(math_file, "r", encoding="utf-8") as f:
                    math_data = json.load(f)
                    for item in math_data:
                        task_info = {
                            "type": "math",
                            "data": item,
                            "rating": item.get("rating", 0),
                            "id": f"math_{hash(str(item))}",
                        }
                        level = min(task_info["rating"], 5)  # Ensure level <= 5
                        self.tasks_by_level[level].append(task_info)
            except Exception as e:
                print(f"Warning: Could not load math tasks: {e}")

        # Load puzzle tasks directly from JSON
        puzzles_file = Path(__file__).parent / "runtimes" / "puzzles.json"
        if puzzles_file.exists():
            try:
                with open(puzzles_file, "r", encoding="utf-8") as f:
                    puzzles_data = json.load(f)

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
                                level = min(task_info["rating"], 5)
                                self.tasks_by_level[level].append(task_info)
                                puzzle_count += 1
            except Exception as e:
                print(f"Warning: Could not load puzzle tasks: {e}")

        # Print summary
        total_tasks = sum(len(tasks) for tasks in self.tasks_by_level.values())
        print(
            f"Loaded {total_tasks} tasks across {len(self.tasks_by_level)} difficulty levels"
        )
        for level in range(0, 6):
            print(f"  Level {level}: {len(self.tasks_by_level[level])} tasks")

    def _get_format_failure_tasks(self) -> List[Task]:
        """Get list of tasks that failed format validation.

        Returns:
            List of Task objects where format reward score is 0.
        """
        format_failures = []
        for task in self.session.tasks.values():
            # Skip tasks that are already reflective (to avoid reflective of reflective)
            if "_reflective" in task.task_id:
                continue

            # Look for format reward in task_rewards
            # Check both format_think and format_answer
            for reward in task.task_rewards:
                if (
                    reward.reward_function_name in ["format_think", "format_answer"]
                    and reward.score == 0
                ):
                    format_failures.append(task)
                    break  # Only add once per task
        return format_failures

    def _get_recent_task_ids(self) -> List[str]:
        """Get recent task base IDs from session history."""
        return [
            tid.rsplit("_", 1)[0] if "_" in tid else tid
            for tid in self.session.task_history
        ]

    def _get_task_counters(self) -> Dict[str, int]:
        """Compute task counters from session data.

        Returns dict mapping task_type to success count:
        +1 for correct tasks (is_correct=True), -1 for failures.
        """
        counters: Dict[str, int] = {}
        for task in self.session.tasks.values():
            task_type = task.task_type
            if task_type not in counters:
                counters[task_type] = 0

            if task.is_correct:
                counters[task_type] += 1
            else:
                counters[task_type] -= 1

        return counters

    def _get_failed_tasks(self) -> Dict[str, str]:
        """Get failed tasks from session data.

        Returns dict mapping task_id to task_name for tasks with is_correct=False.
        """
        return {
            task_id: task.task_name
            for task_id, task in self.session.tasks.items()
            if task.is_correct is False
        }

    def _get_format_failure_tasks(self) -> List[Task]:
        """Get list of tasks that failed format validation (excluding already-reflective tasks).

        Reflective tasks are excluded to prevent cascading reflective prompts.
        Original tasks can be used for reflective learning unlimited times,
        but reflective tasks themselves should not generate further reflective versions.

        Returns:
            List of Task objects where format reward score is 0 and task is not already reflective.
        """
        format_failures = []
        for task in self.session.tasks.values():
            # Skip tasks that are already reflective (to avoid reflective of reflective)
            if "_reflective" in task.task_id:
                continue

            # Look for format reward in task_rewards
            # Check both format_think and format_answer
            for reward in task.task_rewards:
                if (
                    reward.reward_function_name in ["format_think", "format_answer"]
                    and reward.score == 0
                ):
                    format_failures.append(task)
                    break  # Only add once per task
        return format_failures

    def _create_reflective_prompt(self, task: Task) -> str:
        """Create a reflective learning prompt from a failed task.

        Uses task-type-specific formatting (math vs puzzle).

        Args:
            task: The task that failed format validation

        Returns:
            A reflective prompt that guides the model to retry with proper formatting
        """
        if task.task_type == "math":
            return format_reflective_math_prompt(
                original_prompt=task.prompt,
                previous_attempt=task.model_output,
                answer_tag=self.answer_tag,
                think_tag=self.think_tag,
            )
        else:  # puzzle
            return format_reflective_puzzle_prompt(
                original_prompt=task.prompt,
                previous_attempt=task.model_output,
                language=task.language or "python",
                answer_tag=self.answer_tag,
                think_tag=self.think_tag,
            )

    def _get_reflective_prompt(self) -> Optional[Task]:
        """Get a format-failure task wrapped with reflective prompt.

        Key behaviors:
        - Only selects from original (non-reflective) tasks that failed format validation
        - Creates a NEW reflective version each time with an incrementing counter
        - This allows unlimited reflective attempts on the same original task
        - But prevents reflective tasks themselves from generating reflective versions

        Returns:
            A task with reflective prompt if available, None otherwise.
        """
        format_failures = self._get_format_failure_tasks()
        if not format_failures:
            return None

        # Select a random format failure task (guaranteed to be non-reflective)
        selected_task = random.choice(format_failures)

        # Count how many reflective versions already exist for this original task
        reflective_count = sum(
            1
            for task in self.session.tasks.values()
            if task.task_id.startswith(f"{selected_task.task_id}_reflective")
        )

        # Create a reflective version with incrementing counter
        # This allows unlimited reflective attempts on the same original task
        reflective_task = Task(
            task_id=f"{selected_task.task_id}_reflective_{reflective_count}",
            task_name=f"{selected_task.task_name} (reflective attempt {reflective_count + 1})",
            task_type=selected_task.task_type,
            level=selected_task.level,
            prompt=self._create_reflective_prompt(selected_task),
            expected_answer=selected_task.expected_answer,
            language=selected_task.language,
        )

        # Add to session
        self.session.add_task(reflective_task)
        return reflective_task

    def compute_reward(
        self,
        task_id: str,
        model_output: str,
    ) -> float:
        """
        Evaluate model output and update learning state.

        Args:
            task_id: Task identifier
            model_output: Raw model response

        Returns:
            Reward score combining primary correctness and auxiliary metrics
        """
        # Get task from session
        task = self.session.get_task(task_id)
        if task is None:
            raise ValueError(f"Unknown task_id: {task_id}")

        task_type = task.task_type
        expected_output = task.expected_answer

        # Store model_output in task for reward functions
        task.model_output = model_output

        # Get appropriate reward function
        if task_type not in self.reward_functions:
            raise ValueError(f"Unknown task type: {task_type}")

        reward_fn = self.reward_functions[task_type]

        # Compute primary reward
        result = reward_fn.compute_reward(task)
        score = result.score
        primary_info = result.info

        # Create primary reward
        primary_reward = RewardFunctionScore(
            score=score,
            reward_function_name="primary",
            info=primary_info or "",
        )
        task_rewards = [primary_reward]

        # Compute auxiliary rewards and combine them
        is_correct = score == 1  # Threshold for success
        aux_score_dict = self.get_aux_reward_scores(
            model_output, task, is_correct=is_correct
        )

        # Add auxiliary rewards to task_rewards
        for name, data in aux_score_dict.items():
            aux_reward = RewardFunctionScore(
                score=data["score"],
                reward_function_name=name,
                info=data["info"] or "",
            )
            task_rewards.append(aux_reward)

        aux_scores = [(name, data["score"]) for name, data in aux_score_dict.items()]

        # Combine scores (primary + auxiliary)
        if aux_scores:
            aux_avg = sum(s for _, s in aux_scores) / len(aux_scores)
            primary_weight = 1.0 - self.aux_weight
            combined_score = primary_weight * score + self.aux_weight * aux_avg
        else:
            combined_score = score

        # Increment step counter for warmup tracking (BEFORE level update for cooldown check)
        self.global_step += 1

        # GRPO batch accumulation: collect scores until group is complete
        # Extract base task_id (remove instance counter)
        base_task_id = task_id.rsplit("_", 1)[0] if "_" in task_id else task_id
        if base_task_id not in self.grpo_batch_scores:
            self.grpo_batch_scores[base_task_id] = []

        self.grpo_batch_scores[base_task_id].append(combined_score)

        # Check if we have a complete group (GRPO batch size)
        if len(self.grpo_batch_scores[base_task_id]) >= self.num_generations:
            # Complete group: track success at prompt level
            group_scores = self.grpo_batch_scores[base_task_id]
            self._track_success_group(task_type, group_scores)

            # Check if we should advance/demote level
            self._update_level()

            # Clean up batch
            del self.grpo_batch_scores[base_task_id]
        else:
            # Incomplete group: still accumulating responses
            # Don't update level yet, just wait for more responses
            pass

        # Save rewards to session
        self.session.set_reward(task_id, task_rewards, model_output=model_output)

        # Log evaluation if configured
        if self.log_file is not None:
            log_entry = task.to_dict()
            log_entry["timestamp"] = datetime.datetime.now().isoformat()
            log_entry["primary_score"] = primary_reward.score
            log_entry["aux_scores"] = {
                name: data["score"] for name, data in aux_score_dict.items()
            }
            log_entry["combined_score"] = combined_score
            log_entry["info"] = {
                "primary": primary_reward.info,
                **{
                    f"aux_{name}": data["info"] for name, data in aux_score_dict.items()
                },
            }
            with open(self.log_file, "a", encoding="utf-8") as f:
                json.dump(log_entry, f, ensure_ascii=False)
                f.write("\n")

        return combined_score

    def _update_level(self):
        """Update difficulty level based on prompt-level success rate (optimized for GRPO).

        GRPO-aware logic:
        - ADVANCE: Success rate > 80% (high performance at current level)
        - DEMOTE: Success rate < 30% (struggling significantly)
        - Variance check is ONLY used for demotion to prevent thrashing
        - Windows are cleared on level change to start fresh (prevents old data drift)

        Enforces cooldown to prevent cascading changes.
        """
        self._ensure_success_windows_from_session()

        # Check if we're in cooldown period
        steps_since_change = self.global_step - self.last_level_change_step
        if steps_since_change < self.level_change_cooldown:
            return False

        # Minimum samples required (10 prompts = ~40 responses for GRPO)
        min_samples = 10

        # Collect success rates from all tracked task types
        success_rates = []
        for task_type, window in self.success_windows.items():
            if len(window) >= min_samples:
                avg_success = sum(window) / len(window)
                success_rates.append(avg_success)

        if success_rates:
            mean_success_rate = sum(success_rates) / len(success_rates)

            # ADVANCE: High success rate with no variance penalty
            # The model has mastered this difficulty level
            if mean_success_rate > self.success_rate_threshold:
                if self.current_level < 5:
                    print(
                        f"🚀 Advancing to level {self.current_level + 1}: "
                        f"prompt-level success rate = {mean_success_rate:.1%}"
                    )
                    self.current_level = min(self.current_level + 1, 5)
                    self.last_level_change_step = self.global_step
                    # Clear windows to start fresh at new difficulty
                    self.success_windows.clear()
                    return True

            # DEMOTE: Low success rate AND stable (consistent failure)
            # Only demote if we're confident the model is truly stuck
            elif mean_success_rate < self.demote_threshold:
                # Check variance to ensure consistent poor performance
                if len(success_rates) > 1:
                    variance = statistics.variance(success_rates)
                else:
                    window = list(self.success_windows.values())[0]
                    variance = statistics.variance(window) if len(window) > 1 else 0.0

                # Demote only if performance is consistently bad
                if variance < self.variance_threshold:
                    if self.current_level > 0:
                        print(
                            f"📉 Demoting to level {self.current_level - 1}: "
                            f"prompt-level success rate = {mean_success_rate:.1%}, "
                            f"variance = {variance:.4f}"
                        )
                        self.current_level = max(self.current_level - 1, 0)
                        self.last_level_change_step = self.global_step
                        # Clear windows to start fresh at easier difficulty
                        self.success_windows.clear()
                        return True

        return False

    def _track_success(self, task_type: str, is_correct: bool) -> None:
        """Track success/failure in sliding window for a task type (legacy single-response).

        Args:
            task_type: Type of task (e.g., 'math', 'puzzle')
            is_correct: Whether the task was solved correctly
        """
        if task_type not in self.success_windows:
            self.success_windows[task_type] = deque(maxlen=self.window_size)

        # Add 1 for success, 0 for failure
        self.success_windows[task_type].append(1 if is_correct else 0)

    def _track_success_group(self, task_type: str, scores: List[float]) -> None:
        """Track success at prompt-level based on group of GRPO responses.

        For GRPO: considers the prompt "solved" if:
        - ANY response achieves perfect score (1.0), OR
        - Mean score of group >= 0.7 (70% quality threshold)

        This prevents single correct answers from being diluted by incorrect ones,
        and prevents the sliding window from yo-yo'ing on partial solutions.

        Args:
            task_type: Type of task (e.g., 'math', 'puzzle')
            scores: List of scores from GRPO group (typically length 4)
        """
        if task_type not in self.success_windows:
            self.success_windows[task_type] = deque(maxlen=self.window_size)

        # Group success logic: solved if perfect OR high average quality
        mean_score = sum(scores) / len(scores) if scores else 0.0
        max_score = max(scores) if scores else 0.0
        group_success = 1 if (max_score == 1.0 or mean_score >= 0.7) else 0

        self.success_windows[task_type].append(group_success)

    def _ensure_success_windows_from_session(self) -> None:
        """Seed success windows from session history when no tracking data exists."""
        if any(len(window) > 0 for window in self.success_windows.values()):
            return

        for task in self.session.tasks.values():
            if task.is_correct is None:
                continue

            window = self.success_windows.setdefault(
                task.task_type, deque(maxlen=self.window_size)
            )
            window.append(1 if task.is_correct else 0)

    def get_success_rate(self, task_type: Optional[str] = None) -> Dict[str, Any]:
        """Get success rate statistics for task type(s).

        Args:
            task_type: Specific task type to query. If None, returns aggregated stats.

        Returns:
            Dictionary with success_rate, variance, window_size, and samples_count.
        """
        if task_type is not None:
            # Return stats for specific task type
            window = self.success_windows.get(task_type, deque())
            if len(window) == 0:
                return {
                    "task_type": task_type,
                    "success_rate": 0.0,
                    "variance": 0.0,
                    "samples": 0,
                    "window_size": self.window_size,
                }

            success_rate = sum(window) / len(window)
            variance = statistics.variance(window) if len(window) > 1 else 0.0

            return {
                "task_type": task_type,
                "success_rate": success_rate,
                "variance": variance,
                "samples": len(window),
                "window_size": self.window_size,
            }
        else:
            # Return aggregated stats across all task types
            all_stats = []
            for task_type, window in self.success_windows.items():
                if len(window) > 0:
                    success_rate = sum(window) / len(window)
                    variance = statistics.variance(window) if len(window) > 1 else 0.0
                    all_stats.append(
                        {
                            "task_type": task_type,
                            "success_rate": success_rate,
                            "variance": variance,
                            "samples": len(window),
                        }
                    )

            if all_stats:
                mean_success = sum(s["success_rate"] for s in all_stats) / len(
                    all_stats
                )
                mean_variance = sum(s["variance"] for s in all_stats) / len(all_stats)
            else:
                mean_success = 0.0
                mean_variance = 0.0

            return {
                "mean_success_rate": mean_success,
                "mean_variance": mean_variance,
                "threshold_success_rate": self.success_rate_threshold,
                "threshold_variance": self.variance_threshold,
                "samples": sum(s["samples"] for s in all_stats),
                "by_task_type": all_stats,
            }

    def get_prompt(self) -> Optional[Task]:
        """
        Get a task prompt appropriate for current difficulty level.

        With probability reflective_learning_rate (default 0.1), returns a format-failure
        task wrapped in a reflective prompt to help the model learn proper formatting.
        Otherwise, returns a new task based on current difficulty level.

        During warmup stage (first warmup_step iterations), only level 1 tasks are used
        (unless reflective learning is triggered).

        Returns:
            Task object with prompt, expected output, etc., or None if no tasks available.
        """
        # Check for reflective learning trigger (only if not disabled and not in warmup)
        if (
            self.reflective_learning_rate > 0
            and random.random() > (1.0 - self.reflective_learning_rate)
            and not self.is_warmup()
        ):
            reflective_task = self._get_reflective_prompt()
            if reflective_task is not None:
                return reflective_task

        # Normal task selection logic
        # Determine which level to use
        if self.is_warmup():
            # Warmup stage: use level 0 (math tasks only)
            selected_level = 0
        else:
            # Normal curriculum: use current level
            selected_level = self.current_level

        # Get available tasks for selected level
        available_tasks = self.tasks_by_level.get(selected_level, [])

        if not available_tasks:
            # Fallback to any available tasks
            for level in range(0, 6):
                if self.tasks_by_level[level]:
                    available_tasks = self.tasks_by_level[level]
                    break

        if not available_tasks:
            return None

        # Weight against recent tasks
        recent_task_ids = self._get_recent_task_ids()
        weights = []
        for task in available_tasks:
            weight = 1.0
            if task["id"] in recent_task_ids:
                # Reduce weight for recent tasks
                recency_penalty = recent_task_ids.count(task["id"]) / max(
                    len(recent_task_ids), 1
                )
                weight = max(0.1, 1.0 - recency_penalty)
            weights.append(weight)

        # Select task with weighting
        selected_task = random.choices(available_tasks, weights=weights, k=1)[0]

        # Format response based on task type
        if selected_task["type"] == "math":
            # Math task
            math_data = selected_task["data"]
            base_task_id = selected_task["id"]
            # Generate unique task_id for this instance
            unique_task_id = f"{base_task_id}_{self.task_instance_counter}"
            self.task_instance_counter += 1

            task_name = f"math_{math_data.get('prompt', '')[:30]}"
            problem_statement = math_data.get("prompt", "")
            language = math_data.get("lang", "en")
            prompt = format_math_prompt(
                problem_statement, self.answer_tag, language, self.think_tag
            )
            expected_output = math_data.get("response", "")

            task_obj = Task(
                task_id=unique_task_id,
                task_name=task_name,
                task_type="math",
                level=selected_task["rating"],
                prompt=prompt,
                expected_answer=expected_output,
                language=language,
            )
            self.session.add_task(task_obj)
            return task_obj

        elif selected_task["type"] == "puzzle":
            # Puzzle task
            puzzle_data = selected_task["data"]
            base_task_id = selected_task["id"]
            # Generate unique task_id for this instance
            unique_task_id = f"{base_task_id}_{self.task_instance_counter}"
            self.task_instance_counter += 1

            task_name = selected_task["puzzle_name"]
            language = selected_task["language"]  # javascript or python
            prompt = format_puzzle_prompt(
                puzzle_data, language, self.answer_tag, self.think_tag
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
            )
            self.session.add_task(task_obj)
            return task_obj

        return None

    def get_learning_stats(self) -> Dict[str, Any]:
        """Get current learning statistics including sliding window success rates."""
        success_stats = self.get_success_rate()
        return {
            "current_level": self.current_level,
            "task_counters": self._get_task_counters(),
            "failed_tasks_count": len(self._get_failed_tasks()),
            "recent_tasks_count": len(self._get_recent_task_ids()),
            "available_tasks_by_level": {
                level: len(tasks) for level, tasks in self.tasks_by_level.items()
            },
            "aux_reward_functions": list(self.aux_reward_functions.keys()),
            "sliding_window_stats": success_stats,
        }

    def get_aux_reward_scores(
        self,
        model_output: str,
        task: "Task",
        is_correct: bool = False,
    ) -> Dict[str, Dict[str, Any]]:
        """
        Compute all auxiliary reward scores for the given model output.

        Args:
            model_output: Raw model response (already stored in task.model_output)
            task: The Task object (contains expected_answer, language, and model_output)
            is_correct: Whether the primary task was answered correctly

        Returns:
            Dictionary mapping auxiliary reward function names to dicts with 'score' and 'info'
        """
        aux_scores = {}
        for aux_name, aux_fn in self.aux_reward_functions.items():
            try:
                # All auxiliary functions now accept task as first parameter
                aux_result = aux_fn.compute_reward(task, is_correct=is_correct)
                aux_scores[aux_name] = {
                    "score": aux_result.score,
                    "info": aux_result.info,
                }
            except Exception as e:
                print(f"Warning: Auxiliary reward '{aux_name}' failed: {e}")
                aux_scores[aux_name] = {
                    "score": 0.0,
                    "info": str(e),
                }
        return aux_scores

    def is_warmup(self) -> bool:
        """Check if currently in warmup stage."""
        return self.global_step < self.warmup_step
