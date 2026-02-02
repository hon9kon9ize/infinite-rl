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
        aux_weight: float = 0.1,
        llm_judge_weight: float = 0.2,
        use_lang_consistency: bool = False,
        use_repetition: bool = False,
        use_format: bool = True,
        use_reasoning_steps: bool = False,
        use_length: bool = False,
        use_whitespace_collapse: bool = True,
        use_llm_judge: bool = False,
        reasoning_language: str = "en",
        lang_consistency_kwargs: Optional[Dict[str, Any]] = None,
        repetition_kwargs: Optional[Dict[str, Any]] = None,
        format_kwargs: Optional[Dict[str, Any]] = None,
        reasoning_steps_kwargs: Optional[Dict[str, Any]] = None,
        length_kwargs: Optional[Dict[str, Any]] = None,
        whitespace_collapse_kwargs: Optional[Dict[str, Any]] = None,
        llm_judge_kwargs: Optional[Dict[str, Any]] = None,
        log_file: Optional[str] = None,
        window_size: int = 50,
        success_rate_threshold: float = 0.7,
        variance_threshold: float = 0.15,
        demote_threshold: float = 0.4,
        warmup_step: int = 32,
        reflective_learning_rate: float = 0.2,
        level_change_cooldown: int = 5,
        num_generations: int = 4,
        puzzle_one_shot: bool = False,
    ):
        """
        Initialize curriculum learning.

        Args:
            timeout: Timeout for reward function execution
            answer_tag: Tag used to extract answers from model responses
            think_tag: Tag used to extract reasoning from model responses
            aux_weight: Weight for auxiliary rewards in combined score (0-1, default: 0.1)
            llm_judge_weight: Weight for LLM Judge reward, computed independently of format/correctness gates (0-1, default: 0.2)
            use_lang_consistency: Enable language consistency auxiliary reward
            use_repetition: Enable repetition penalty auxiliary reward
            use_format: Enable format validation auxiliary reward
            use_reasoning_steps: Enable chain-of-thought reasoning steps bonus
            use_length: Enable response length regularizer
            use_whitespace_collapse: Enable whitespace collapse detector (default: True)
            use_llm_judge: Enable LLM-based quality evaluation via remote sglang server (default: False)
            reasoning_language: ISO language code for reasoning analysis (default: "en")
            lang_consistency_kwargs: Keyword arguments for LangConsistencyRewardFunction
            repetition_kwargs: Keyword arguments for RepetitionRewardFunction
            format_kwargs: Keyword arguments for FormatRewardFunction
            reasoning_steps_kwargs: Keyword arguments for ReasoningStepsRewardFunction
            length_kwargs: Keyword arguments for LengthRewardFunction
            whitespace_collapse_kwargs: Keyword arguments for WhitespaceCollapseRewardFunction
            llm_judge_kwargs: Keyword arguments for LLMJudgeRewardFunction (api_host, api_port, model_name, etc.)
            log_file: Path to the logging file (JSON Lines format). If None, defaults to 'curriculum_log.jsonl' in the module directory.
            window_size: Size of the sliding window for success rate tracking (default: 50)
            success_rate_threshold: Required success rate for difficulty increase (default: 0.7 = 70%)
            variance_threshold: Maximum variance for success rate stability (default: 0.15)
            demote_threshold: Success rate threshold for difficulty decrease (default: 0.4 = 40%)
            warmup_step: Number of initial steps to only use level 0 tasks (default: 32)
            reflective_learning_rate: Probability of triggering reflective learning on format failures (default: 0.1). Set to 0 to disable.
            level_change_cooldown: Minimum steps between level changes to prevent rapid fluctuations (default: 5)
            num_generations: Number of generations per prompt for GRPO batching (default: 4)
            puzzle_one_shot: Whether to include one-shot examples in puzzle prompts (default: False)
        """
        self.timeout = timeout
        self.answer_tag = answer_tag
        self.think_tag = think_tag
        self.puzzle_one_shot = puzzle_one_shot
        self.aux_weight = aux_weight
        self.llm_judge_weight = llm_judge_weight
        self.reasoning_language = reasoning_language
        self.reward_functions = get_reward_functions(
            timeout=timeout, answer_tag=answer_tag, think_tag=think_tag
        )

        # Auxiliary reward functions configuration
        self.use_lang_consistency = use_lang_consistency
        self.use_repetition = use_repetition
        self.use_format = use_format
        self.use_reasoning_steps = use_reasoning_steps
        self.use_length = use_length
        self.use_whitespace_collapse = use_whitespace_collapse
        self.use_llm_judge = use_llm_judge
        self.lang_consistency_kwargs = lang_consistency_kwargs or {}
        self.repetition_kwargs = repetition_kwargs or {}
        self.format_kwargs = format_kwargs or {}
        self.reasoning_steps_kwargs = reasoning_steps_kwargs or {}
        self.length_kwargs = length_kwargs or {}
        self.whitespace_collapse_kwargs = whitespace_collapse_kwargs or {}
        self.llm_judge_kwargs = llm_judge_kwargs or {}

        # Validate LLM Judge configuration if enabled
        if self.use_llm_judge:
            if "api_host" not in self.llm_judge_kwargs:
                raise ValueError(
                    "use_llm_judge=True requires 'api_host' in llm_judge_kwargs"
                )
            if "api_port" not in self.llm_judge_kwargs:
                raise ValueError(
                    "use_llm_judge=True requires 'api_port' in llm_judge_kwargs"
                )
            if "model_name" not in self.llm_judge_kwargs:
                raise ValueError(
                    "use_llm_judge=True requires 'model_name' in llm_judge_kwargs"
                )

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
        # Deques to track success/failure per level (not task type)
        # Maps level (int) -> deque of 0s/1s
        self.success_windows: Dict[int, deque] = {}

        # Level change cooldown: prevent successive advancements too quickly
        self.last_level_change_step: int = -999  # Track when last level change occurred
        self.level_change_cooldown: int = level_change_cooldown

        # GRPO batch tracking: accumulate group responses until prompt-level decision
        self.grpo_batch_primary_scores: Dict[str, List[float]] = (
            {}
        )  # Maps task_id -> list of primary scores (for curriculum)
        self.grpo_batch_outputs: Dict[str, List[str]] = (
            {}
        )  # Maps task_id -> list of model outputs (for logging all generations)
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
                        target_language=self.reasoning_language,
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

        if self.use_whitespace_collapse:
            try:
                from .reward_functions import WhitespaceCollapseRewardFunction

                self.aux_reward_functions["whitespace_collapse"] = (
                    WhitespaceCollapseRewardFunction(
                        task_name="whitespace_collapse",
                        timeout=self.timeout,
                        answer_tag=self.answer_tag,
                        think_tag=self.think_tag,
                        reasoning_language=self.reasoning_language,
                        **self.whitespace_collapse_kwargs,
                    )
                )
            except Exception as e:
                print(
                    f"Warning: Could not initialize WhitespaceCollapseRewardFunction: {e}"
                )

        if self.use_llm_judge:
            try:
                from .reward_functions import LLMJudgeRewardFunction

                self.aux_reward_functions["llm_judge"] = LLMJudgeRewardFunction(
                    task_name="llm_judge",
                    timeout=self.timeout,
                    answer_tag=self.answer_tag,
                    think_tag=self.think_tag,
                    **self.llm_judge_kwargs,
                )
            except Exception as e:
                print(f"Warning: Could not initialize LLMJudgeRewardFunction: {e}")

    def _load_available_tasks(self):
        """Load all available tasks and their ratings."""
        self.tasks_by_level: Dict[int, List[Dict[str, Any]]] = {
            i: [] for i in range(0, 7)  # 0-6 level
        }

        # Helper function to load JSON file from package resources
        def load_runtime_json(filename):
            """Load JSON file from runtime resources, trying multiple methods."""
            try:
                # Method 1: Try importlib.resources (Python 3.9+)
                try:
                    from importlib import resources

                    try:
                        # Python 3.9+ API
                        data_text = (
                            resources.files("infinite_rl.runtimes")
                            .joinpath(filename)
                            .read_text(encoding="utf-8")
                        )
                        print(
                            f"DEBUG: Loaded {filename} via importlib.resources (Python 3.9+)"
                        )
                        return json.loads(data_text)
                    except AttributeError:
                        # Python 3.7-3.8 API
                        data_text = resources.read_text(
                            "infinite_rl.runtimes", filename, encoding="utf-8"
                        )
                        print(
                            f"DEBUG: Loaded {filename} via importlib.resources (Python 3.7-3.8)"
                        )
                        return json.loads(data_text)
                except Exception as resources_error:
                    print(
                        f"DEBUG: importlib.resources failed for {filename}: {resources_error}"
                    )

                # Method 2: Fallback to Path-based loading
                file_path = Path(__file__).parent / "runtimes" / filename
                print(f"DEBUG: Trying Path-based loading: {file_path}")
                print(f"DEBUG: File exists: {file_path.exists()}")
                if file_path.exists():
                    with open(file_path, "r", encoding="utf-8") as f:
                        data = json.load(f)
                        print(f"DEBUG: Loaded {filename} via Path-based loading")
                        return data
                else:
                    print(f"DEBUG: {filename} not found at {file_path}")
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
                print(f"DEBUG: Processing {len(math_data)} math items")
                for item in math_data:
                    task_info = {
                        "type": "math",
                        "data": item,
                        "rating": item.get("rating", 0),
                        "id": f"math_{hash(str(item))}",
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
                print(
                    f"DEBUG: Processing puzzles data, keys: {puzzles_data.keys() if isinstance(puzzles_data, dict) else 'not a dict'}"
                )

                for lang in ["javascript", "python"]:
                    if lang in puzzles_data:
                        puzzles_list = puzzles_data[lang]
                        print(f"DEBUG: Found {len(puzzles_list)} {lang} puzzles")
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
                        print(f"DEBUG: Added {puzzle_count} {lang} puzzles to tasks")
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

                print(f"DEBUG: Found {len(truthy_list)} truthy tasks")
                truthy_count = 0

                for idx, truthy_item in enumerate(truthy_list):
                    if isinstance(truthy_item, dict) and "prompt" in truthy_item:
                        task_info = {
                            "type": "truthy",
                            "data": truthy_item,
                            "rating": None,  # Truthy tasks not limited by rating
                            "id": f"truthy_{idx}_{truthy_item.get('id', '')}",
                        }
                        # Distribute truthy tasks across all levels
                        # Each level gets all truthy tasks with 20% weight
                        for level in range(0, 7):
                            self.tasks_by_level[level].append(task_info)
                        truthy_count += 1

                print(f"DEBUG: Added {truthy_count} truthy tasks to all levels")
            except Exception as e:
                print(f"Warning: Could not process truthy tasks: {e}")
                import traceback

                traceback.print_exc()

        # Print summary
        total_tasks = sum(len(tasks) for tasks in self.tasks_by_level.values())
        print(
            f"Loaded {total_tasks} tasks across {len(self.tasks_by_level)} difficulty levels"
        )
        for level in range(0, 7):
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

        # Store model_output in task for reward functions
        task.model_output = model_output

        # FORMAT AS HARD GATE: Check format validity before evaluating primary task
        # format_valid is True only when BOTH think and answer tags are properly formatted
        format_valid = True
        format_failure_reason = ""

        if self.use_format:
            # Check both think and answer format - BOTH must pass for format_valid=True
            format_think_valid = True
            format_answer_valid = True

            if "format_think" in self.aux_reward_functions:
                format_fn = self.aux_reward_functions["format_think"]
                format_result = format_fn.compute_reward(task, is_correct=False)
                # Format functions return 1.0 for success, -1.0 for failure
                if format_result.score < 1.0:
                    format_think_valid = False
                    format_failure_reason = format_result.info

            if "format_answer" in self.aux_reward_functions:
                format_fn = self.aux_reward_functions["format_answer"]
                format_result = format_fn.compute_reward(task, is_correct=False)
                # Format functions return 1.0 for success, -1.0 for failure
                if format_result.score < 1.0:
                    format_answer_valid = False
                    if not format_failure_reason:
                        format_failure_reason = format_result.info

            # format_valid is True only if BOTH think and answer tags are valid
            format_valid = format_think_valid and format_answer_valid

        # Initialize aux_score_dict (will be computed based on outcome)
        aux_score_dict = {}

        # Handle truthy tasks: primary score comes from llm_judge
        if task_type == "truthy":
            if "llm_judge" not in self.aux_reward_functions:
                raise ValueError(
                    "Truthy tasks require llm_judge. Please set use_llm_judge=True with api_host, api_port, and model_name"
                )
            try:
                llm_judge_result = self.aux_reward_functions[
                    "llm_judge"
                ].compute_reward(task, is_correct=False)
                primary_score = llm_judge_result.score
                primary_info = llm_judge_result.info
            except Exception as e:
                print(f"Error: LLM Judge failed for truthy task: {e}")
                primary_score = 0.0
                primary_info = f"LLM Judge error: {e}"
        else:
            # Get appropriate reward function for non-truthy tasks
            if task_type not in self.reward_functions:
                raise ValueError(f"Unknown task type: {task_type}")

            reward_fn = self.reward_functions[task_type]

            # Compute primary reward (regardless of format for leaky gate logic)
            result = reward_fn.compute_reward(task)
            primary_score = result.score
            primary_info = result.info

        # COMPUTE LLM JUDGE INDEPENDENTLY: For non-truthy tasks, compute as auxiliary
        if task_type != "truthy" and "llm_judge" in self.aux_reward_functions:
            try:
                llm_judge_result = self.aux_reward_functions[
                    "llm_judge"
                ].compute_reward(task, is_correct=False)
                aux_score_dict["llm_judge"] = {
                    "score": llm_judge_result.score,
                    "info": llm_judge_result.info,
                }
            except Exception as e:
                print(f"Warning: LLM Judge reward failed: {e}")
                aux_score_dict["llm_judge"] = {
                    "score": 0.0,
                    "info": str(e),
                }

        # STRICT GATE: For non-truthy tasks, require both format validity AND correctness
        # For truthy tasks, use the primary score directly (continuous quality score from judge)
        # No partial credit for non-truthy - must have both tags AND correct answer

        if task_type == "truthy":
            # Truthy tasks: use primary score directly (no strict gate)
            primary_reward = RewardFunctionScore(
                score=primary_score,
                reward_function_name="primary",
                info=primary_info or "",
            )
            task_rewards = [primary_reward]

            # Compute auxiliary rewards for truthy tasks (format doesn't apply)
            other_aux_scores = self.get_aux_reward_scores(
                model_output, task, is_correct=False
            )

            # Add all auxiliary rewards (no capping for truthy)
            for name, data in other_aux_scores.items():
                if name != "llm_judge":  # llm_judge is already the primary
                    aux_reward = RewardFunctionScore(
                        score=data["score"],
                        reward_function_name=name,
                        info=data["info"] or "",
                    )
                    task_rewards.append(aux_reward)
                    aux_score_dict[name] = data

            score = primary_score
            is_correct = (
                primary_score > 0.7
            )  # Consider score > 0.7 as success for tracking
        else:
            # Non-truthy tasks: apply strict gate
            is_correct = primary_score == 1  # Threshold for success

            if format_valid and is_correct:
                # Both gates passed - format is valid AND answer is correct
                primary_reward = RewardFunctionScore(
                    score=primary_score,
                    reward_function_name="primary",
                    info=primary_info or "",
                )
                task_rewards = [primary_reward]

                # Compute auxiliary rewards and combine them
                other_aux_scores = self.get_aux_reward_scores(
                    model_output, task, is_correct=is_correct
                )

                # Add non-llm_judge auxiliary rewards to task_rewards
                for name, data in other_aux_scores.items():
                    if name != "llm_judge":  # llm_judge already added above
                        aux_reward = RewardFunctionScore(
                            score=data["score"],
                            reward_function_name=name,
                            info=data["info"] or "",
                        )
                        task_rewards.append(aux_reward)
                        aux_score_dict[name] = data

                score = primary_score
            else:
                # Failed gates: either format invalid OR answer incorrect
                primary_reward = RewardFunctionScore(
                    score=primary_score,
                    reward_function_name="primary",
                    info=primary_info or "",
                )
                task_rewards = [primary_reward]

                # Add format feedback if applicable
                if not format_valid:
                    task_rewards.append(
                        RewardFunctionScore(
                            score=0.0,
                            reward_function_name="format",
                            info=f"Format validation failed: {format_failure_reason}",
                        )
                    )

                # Still compute aux scores for logging, but cap positive ones
                other_aux_scores = self.get_aux_reward_scores(
                    model_output, task, is_correct=False
                )

                # Cap positive auxiliary rewards when primary task failed
                # But NOT for llm_judge, which is already separate
                capped_aux_scores = {
                    name: {
                        "score": min(0.0, data["score"]),
                        "info": (
                            data["info"]
                            if data["score"] < 0
                            else "Primary task failed, positive auxiliary reward capped at 0"
                        ),
                    }
                    for name, data in other_aux_scores.items()
                    if name != "llm_judge"  # llm_judge not capped
                }

                # Add auxiliary rewards to task_rewards
                for name, data in capped_aux_scores.items():
                    aux_reward = RewardFunctionScore(
                        score=data["score"],
                        reward_function_name=name,
                        info=data["info"] or "",
                    )
                    task_rewards.append(aux_reward)
                    aux_score_dict[name] = data

                score = 0.0
                is_correct = False

        # GRPO batch accumulation: collect scores until group is complete
        # Extract base task_id (remove instance counter)
        base_task_id = task_id.rsplit("_", 1)[0] if "_" in task_id else task_id
        if base_task_id not in self.grpo_batch_primary_scores:
            self.grpo_batch_primary_scores[base_task_id] = []
            self.grpo_batch_outputs[base_task_id] = []

        self.grpo_batch_primary_scores[base_task_id].append(score)
        self.grpo_batch_outputs[base_task_id].append(model_output)

        # Check if we have a complete group (GRPO batch size)
        if len(self.grpo_batch_primary_scores[base_task_id]) >= self.num_generations:
            # Complete group: track success at prompt level
            primary_scores = self.grpo_batch_primary_scores[base_task_id]
            self._track_success_group(task.level, primary_scores)

            # Increment step counter ONCE per complete prompt group
            self.global_step += 1

            # Check if we should advance/demote level
            self._update_level()

            # Log evaluation if configured (only when batch is complete)
            if self.log_file is not None:
                log_entry = task.to_dict()
                log_entry["timestamp"] = datetime.datetime.now().isoformat()
                log_entry["primary_score"] = primary_reward.score
                log_entry["aux_scores"] = {
                    name: data["score"] for name, data in aux_score_dict.items()
                }
                log_entry["info"] = {
                    "primary": primary_reward.info,
                    **{
                        f"aux_{name}": data["info"]
                        for name, data in aux_score_dict.items()
                    },
                }
                # Add GRPO batch information
                log_entry["grpo_batch_size"] = len(primary_scores)
                log_entry["grpo_primary_scores"] = primary_scores
                log_entry["grpo_model_outputs"] = self.grpo_batch_outputs[base_task_id]
                with open(self.log_file, "a", encoding="utf-8") as f:
                    json.dump(log_entry, f, ensure_ascii=False)
                    f.write("\n")

            # Clean up completed batch
            del self.grpo_batch_primary_scores[base_task_id]
            del self.grpo_batch_outputs[base_task_id]
        elif self.num_generations == 1:
            # Single evaluation mode (non-GRPO): log immediately
            self._track_success_group(task.level, [score])
            self.global_step += 1
            self._update_level()

            # Log evaluation if configured
            if self.log_file is not None:
                log_entry = task.to_dict()
                log_entry["timestamp"] = datetime.datetime.now().isoformat()
                log_entry["primary_score"] = primary_reward.score
                log_entry["aux_scores"] = {
                    name: data["score"] for name, data in aux_score_dict.items()
                }
                log_entry["info"] = {
                    "primary": primary_reward.info,
                    **{
                        f"aux_{name}": data["info"]
                        for name, data in aux_score_dict.items()
                    },
                }
                with open(self.log_file, "a", encoding="utf-8") as f:
                    json.dump(log_entry, f, ensure_ascii=False)
                    f.write("\n")
        elif len(self.grpo_batch_primary_scores[base_task_id]) > self.num_generations:
            # Safety: If we somehow got MORE than expected, process anyway
            print(
                f"Warning: Task {base_task_id} has {len(self.grpo_batch_primary_scores[base_task_id])} responses "
                f"(expected {self.num_generations}). Processing batch anyway."
            )
            primary_scores = self.grpo_batch_primary_scores[base_task_id][
                : self.num_generations
            ]
            self._track_success_group(task.level, primary_scores)
            self.global_step += 1
            self._update_level()

            # Log evaluation if configured (only when batch is complete)
            if self.log_file is not None:
                log_entry = task.to_dict()
                log_entry["timestamp"] = datetime.datetime.now().isoformat()
                log_entry["primary_score"] = primary_reward.score
                log_entry["aux_scores"] = {
                    name: data["score"] for name, data in aux_score_dict.items()
                }
                log_entry["info"] = {
                    "primary": primary_reward.info,
                    **{
                        f"aux_{name}": data["info"]
                        for name, data in aux_score_dict.items()
                    },
                }
                # Add GRPO batch information
                log_entry["grpo_batch_size"] = len(primary_scores)
                log_entry["grpo_primary_scores"] = primary_scores
                with open(self.log_file, "a", encoding="utf-8") as f:
                    json.dump(log_entry, f, ensure_ascii=False)
                    f.write("\n")

            del self.grpo_batch_primary_scores[base_task_id]
            del self.grpo_batch_outputs[base_task_id]
        else:
            # Incomplete group: still accumulating responses
            # Don't update level yet, just wait for more responses
            pass

        # Periodic cleanup of stale batches (prevents memory leak if training crashes)
        if self.global_step % 100 == 0 and len(self.grpo_batch_primary_scores) > 50:
            print(
                f"Warning: {len(self.grpo_batch_primary_scores)} incomplete GRPO batches detected. "
                f"This may indicate a problem with the training loop. Cleaning up stale batches..."
            )
            # Keep only the most recent 20 incomplete batches
            if len(self.grpo_batch_primary_scores) > 20:
                sorted_batches = sorted(self.grpo_batch_primary_scores.items())
                stale_count = len(sorted_batches) - 20
                for batch_id, _ in sorted_batches[:stale_count]:
                    # Only delete if key exists (may not exist if batch was created in test/externally)
                    if batch_id in self.grpo_batch_primary_scores:
                        del self.grpo_batch_primary_scores[batch_id]
                    if batch_id in self.grpo_batch_outputs:
                        del self.grpo_batch_outputs[batch_id]
                print(f"Cleaned up {stale_count} stale batches")

        # Save rewards to session
        self.session.set_reward(task_id, task_rewards, model_output=model_output)

        return score

    def _update_level(self):
        """Update difficulty level based on prompt-level success rate (optimized for GRPO).

        GRPO-aware logic:
        - ADVANCE: Success rate > threshold (high performance at CURRENT level only)
        - DEMOTE: Success rate < demote_threshold (struggling significantly at current level)
        - Variance check is ONLY used for demotion to prevent thrashing
        - Windows are cleared on level change to start fresh (prevents old data drift)
        - Only checks current level's success rate to avoid leveling up too fast

        Enforces cooldown to prevent cascading changes.
        """
        self._ensure_success_windows_from_session()

        # Check if we're in cooldown period
        steps_since_change = self.global_step - self.last_level_change_step
        if steps_since_change < self.level_change_cooldown:
            return False

        # Minimum samples required (10 prompts = ~40 responses for GRPO)
        min_samples = 10

        # Only check CURRENT level's success rate
        current_window = self.success_windows.get(self.current_level)
        if current_window is None or len(current_window) < min_samples:
            return False

        # Calculate success rate for current level only
        current_success_rate = sum(current_window) / len(current_window)
        variance = (
            statistics.variance(current_window) if len(current_window) > 1 else 0.0
        )

        # ADVANCE: High success rate at current level
        # The model has mastered this difficulty level
        if current_success_rate > self.success_rate_threshold:
            if self.current_level < 6:
                print(
                    f"🚀 Advancing to level {self.current_level + 1}: "
                    f"level {self.current_level} success rate = {current_success_rate:.1%}"
                )
                self.current_level = min(self.current_level + 1, 6)
                self.last_level_change_step = self.global_step
                # Clear windows to start fresh at new difficulty
                self.success_windows.clear()
                return True

        # DEMOTE: Low success rate at current level AND stable (consistent failure)
        # Only demote if we're confident the model is truly stuck
        elif current_success_rate < self.demote_threshold:
            # Demote only if performance is consistently bad (low variance)
            if variance < self.variance_threshold:
                if self.current_level > 0:
                    print(
                        f"📉 Demoting to level {self.current_level - 1}: "
                        f"level {self.current_level} success rate = {current_success_rate:.1%}, "
                        f"variance = {variance:.4f}"
                    )
                    self.current_level = max(self.current_level - 1, 0)
                    self.last_level_change_step = self.global_step
                    # Clear windows to start fresh at easier difficulty
                    self.success_windows.clear()
                    return True

        return False

    def _track_success(self, level: int, is_correct: bool) -> None:
        """Track success/failure in sliding window for a level (legacy single-response).

        Args:
            level: Difficulty level of the task (0-6)
            is_correct: Whether the task was solved correctly
        """
        if level not in self.success_windows:
            self.success_windows[level] = deque(maxlen=self.window_size)

        # Add 1 for success, 0 for failure
        self.success_windows[level].append(1 if is_correct else 0)

    def _track_success_group(self, level: int, primary_scores: List[float]) -> None:
        """Track success at prompt-level based on group of GRPO responses.

        For GRPO: considers the prompt "solved" if:
        - ANY response achieves perfect PRIMARY score (1.0), OR
        - Mean PRIMARY score >= 0.7 (70% correctness threshold)

        Uses primary scores (task correctness) for curriculum progression.
        This separates task-solving ability from formatting/quality issues.

        Args:
            level: Difficulty level of the task (0-6)
            primary_scores: List of primary correctness scores (for curriculum)
        """
        if level not in self.success_windows:
            self.success_windows[level] = deque(maxlen=self.window_size)

        # Group success logic: use PRIMARY scores for curriculum decisions
        # This separates task-solving ability from formatting/quality
        mean_primary = (
            sum(primary_scores) / len(primary_scores) if primary_scores else 0.0
        )
        max_primary = max(primary_scores) if primary_scores else 0.0
        group_success = 1 if (max_primary == 1.0 or mean_primary >= 0.7) else 0

        self.success_windows[level].append(group_success)

    def _ensure_success_windows_from_session(self) -> None:
        """Seed success windows from session history when no tracking data exists."""
        if any(len(window) > 0 for window in self.success_windows.values()):
            return

        for task in self.session.tasks.values():
            if task.is_correct is None:
                continue

            window = self.success_windows.setdefault(
                task.level, deque(maxlen=self.window_size)
            )
            window.append(1 if task.is_correct else 0)

    def get_success_rate(self, level: Optional[int] = None) -> Dict[str, Any]:
        """Get success rate statistics for level(s).

        Args:
            level: Specific difficulty level to query. If None, returns aggregated stats.

        Returns:
            Dictionary with success_rate, variance, window_size, and samples_count.
        """
        if level is not None:
            # Return stats for specific level
            window = self.success_windows.get(level, deque())
            if len(window) == 0:
                return {
                    "level": level,
                    "success_rate": 0.0,
                    "variance": 0.0,
                    "samples": 0,
                    "window_size": self.window_size,
                }

            success_rate = sum(window) / len(window)
            variance = statistics.variance(window) if len(window) > 1 else 0.0

            return {
                "level": level,
                "success_rate": success_rate,
                "variance": variance,
                "samples": len(window),
                "window_size": self.window_size,
            }
        else:
            # Return aggregated stats across all levels
            all_stats = []
            for level, window in self.success_windows.items():
                if len(window) > 0:
                    success_rate = sum(window) / len(window)
                    variance = statistics.variance(window) if len(window) > 1 else 0.0
                    all_stats.append(
                        {
                            "level": level,
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
                "by_level": all_stats,
            }

    def get_prompt(self) -> Optional[Task]:
        """
        Get a task prompt appropriate for current difficulty level.

        With probability reflective_learning_rate (default 0.1), returns a format-failure
        task wrapped in a reflective prompt to help the model learn proper formatting.
        Otherwise, returns a new task based on current difficulty level.

        During warmup stage (first warmup_step iterations), only level 0 tasks are used
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
            for level in range(0, 7):
                if self.tasks_by_level[level]:
                    available_tasks = self.tasks_by_level[level]
                    break

        if not available_tasks:
            return None

        # Weight against recent tasks and apply truthy 20% weight
        recent_task_ids = self._get_recent_task_ids()
        weights = []
        for task in available_tasks:
            # Base weight: 1.0 for regular tasks, 0.2 for truthy tasks
            if task["type"] == "truthy":
                weight = 0.2  # Truthy tasks: 20% weight
            else:
                weight = 1.0  # Math/Puzzle tasks: 80% weight

            # Reduce weight for recent tasks
            if task["id"] in recent_task_ids:
                recency_penalty = recent_task_ids.count(task["id"]) / max(
                    len(recent_task_ids), 1
                )
                weight = max(0.1, weight * (1.0 - recency_penalty))
            weights.append(weight)

        # Select task with weighting
        selected_task = random.choices(available_tasks, weights=weights, k=1)[0]

        # Format response based on task type
        if selected_task["type"] == "truthy":
            # Truthy task
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

            # Format the prompt with chosen and rejected options
            full_prompt = (
                f"{user_prompt}\n\n## Chosen: {chosen}\n\n## Rejected: {rejected}"
            )

            # Expected answer is a conversation format
            expected_output = {
                "type": "truthy",
                "conversation": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": full_prompt},
                    # model_output will be added as assistant message during evaluation
                ],
            }

            task_obj = Task(
                task_id=unique_task_id,
                task_name=task_name,
                task_type="truthy",
                level=random.randint(
                    0, 6
                ),  # Random level since truthy not limited by rating
                prompt=full_prompt,
                expected_answer=expected_output,
                language="multi",  # Truthy can be multilingual
                reasoning_language=truthy_data.get("language", "en"),
            )
            self.session.add_task(task_obj)
            return task_obj

        elif selected_task["type"] == "math":
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

    def get_rewards(self, task_ids: List[str]) -> List[float]:
        """Calculate combined reward scores for multiple completed tasks.

        Retrieves primary correctness and auxiliary scores from task_rewards,
        normalizes each auxiliary score to [0, 1] range, then applies aux_weight
        to blend primary with average normalized auxiliary scores.

        Normalization strategy:
        - Primary score: already binary (0 or 1)
        - Auxiliary scores: clip to [-1, 1], then shift to [0, 1]
          (assumes auxiliary rewards designed to be in [-1, 1] range)
        - Combined: primary_weight * primary + aux_weight * avg(normalized_aux)

        Args:
            task_ids: List of task identifiers (should have been processed by compute_reward)

        Returns:
            List of combined reward scores in the range [0, 1] suitable for RL training

        Raises:
            ValueError: If any task_id not found in session
        """
        combined_rewards = []

        # Precompute the judge score to get the max score and min score to normalize
        highest_judge_score = -float("inf")
        lowest_judge_score = float("inf")

        # Only scan judge scores if llm_judge is enabled
        if self.use_llm_judge:
            for task_id in task_ids:
                task = self.session.get_task(task_id)
                if task is None:
                    raise ValueError(f"Unknown task_id: {task_id}")

                for reward in task.task_rewards:
                    if reward.reward_function_name == "llm_judge":
                        if reward.score > highest_judge_score:
                            highest_judge_score = reward.score
                        if reward.score < lowest_judge_score:
                            lowest_judge_score = reward.score
            # Ensure min <= max (if no judge scores exist, set default range)
            if highest_judge_score == -float("inf"):
                highest_judge_score = 1e-6
            if lowest_judge_score == float("inf"):
                lowest_judge_score = -1e-6

        for task_id in task_ids:
            task = self.session.get_task(task_id)
            if task is None:
                raise ValueError(f"Unknown task_id: {task_id}")

            # Extract primary and auxiliary scores from task_rewards
            primary_score = None
            judge_score = 0.0
            aux_scores = {}
            is_truthy_task = task.task_type == "truthy"

            for reward in task.task_rewards:
                if reward.reward_function_name == "primary":
                    primary_score = reward.score
                elif reward.reward_function_name == "llm_judge":
                    # For truthy tasks, primary score IS llm_judge, don't count as auxiliary
                    if not is_truthy_task:
                        judge_score = self._normalize_score(
                            reward.score,
                            lowest_judge_score,
                            highest_judge_score,
                        )
                elif reward.reward_function_name not in ["format"]:
                    # Collect all auxiliary scores (format is a hard gate, not blended)
                    aux_scores[reward.reward_function_name] = reward.score

            if primary_score is None:
                primary_score = 0.0

            # Normalize auxiliary scores to [0, 1]
            if aux_scores:
                normalized_aux_list = []
                for aux_name, aux_value in aux_scores.items():
                    # Clip to [-1, 1] range (auxiliary rewards designed to be in this range)
                    clipped = max(-1.0, min(1.0, aux_value))
                    # Shift from [-1, 1] to [0, 1]
                    normalized = (clipped + 1.0) / 2.0
                    normalized_aux_list.append(normalized)

                aux_avg = sum(normalized_aux_list) / len(normalized_aux_list)
            else:
                # No auxiliary scores: use middle value
                aux_avg = 0.5

            # Blend primary and normalized auxiliary using aux_weight
            # Weight calculation depends on task type and whether llm_judge is enabled
            if is_truthy_task:
                # For truthy tasks, primary score IS judge score
                primary_weight = 1.0 - self.aux_weight
            else:
                # For non-truthy tasks, only subtract llm_judge_weight if enabled
                if self.use_llm_judge:
                    primary_weight = 1.0 - self.aux_weight - self.llm_judge_weight
                else:
                    primary_weight = 1.0 - self.aux_weight

            # Combine scores: only include judge term if llm_judge is enabled
            judge_contribution = (
                self.llm_judge_weight * judge_score
                if (self.use_llm_judge and not is_truthy_task)
                else 0.0
            )
            combined_score = (
                (primary_weight * primary_score)
                + (self.aux_weight * aux_avg)
                + judge_contribution
            )

            # Ensure final score is in [0, 1]
            combined_score = max(0.0, min(1.0, combined_score))
            combined_rewards.append(combined_score)

        return combined_rewards

    def _normalize_score(
        self,
        raw_score: float,
        min_score: float,
        max_score: float,
    ) -> float:
        """Normalize raw score to [0, 1] based on observed min/max scores.

        Args:
            raw_score: The raw score to normalize
            min_score: The minimum observed score
            max_score: The maximum observed score

        Returns:
            Normalized score in [0, 1]
        """
        if max_score - min_score < 1e-6:
            return 0.0  # Avoid division by zero if no range

        normalized = (raw_score - min_score) / (max_score - min_score)
        return max(0.0, min(1.0, normalized))

    def is_warmup(self) -> bool:
        """Check if currently in warmup stage."""
        return self.global_step < self.warmup_step
