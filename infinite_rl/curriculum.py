"""
Curriculum Learning for Infinite RL.

This module provides curriculum learning functionality that progressively
increases task difficulty based on model performance using a sliding window
success rate with variance tracking.
"""

import json
import random
from collections import deque
from typing import Dict, List, Optional, Any
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
    format_truthy_judge_system_prompt,
    format_truthy_user_prompt,
)


class CurriculumLearning:

    def __init__(
        self,
        session: Optional[Session] = None,
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
        truthy_learning_rate: float = 0.2,
        level_change_cooldown: int = 5,
        num_generations: int = 4,
        puzzle_one_shot: bool = False,
    ):
        """
        Initialize curriculum learning.

        Args:
            session: Session object for task management. If None, creates a new session.
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
            reflective_learning_rate: Probability of triggering reflective learning on format failures (default: 0.2). Set to 0 to disable.
            truthy_learning_rate: Probability of including truthy tasks in the curriculum (default: 0.2). Set to 0 to disable.
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

        # Truthy learning configuration
        self.truthy_learning_rate = truthy_learning_rate

        # Session management
        self.session = session or Session()
        self.log_file = Path(log_file) if log_file is not None else None

        # Track which tasks have been logged to prevent duplicate logging
        self.logged_tasks: set = set()

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

        self.num_generations: int = (
            num_generations  # Number of generations per prompt (configurable)
        )

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

            # Look for format reward in latest generation
            # Check both format_think and format_answer
            latest_gen = task.latest_generation
            if latest_gen:
                for reward in latest_gen.rewards:
                    if (
                        reward.reward_function_name in ["format_think", "format_answer"]
                        and reward.score == 0
                    ):
                        format_failures.append(task)
                        break  # Only add once per task
        return format_failures

    def compute_reward(
        self,
        task_id: str,
        model_output: str,
    ) -> float:
        """
        Evaluate model output and update learning state.

        Dispatches to task-type-specific handlers:
        - Truthy tasks: LLM Judge as primary score
        - Math/Puzzle: Primary task correctness with format gates

        Args:
            task_id: Task identifier
            model_output: Raw model response

        Returns:
            Reward score (for curriculum tracking and GRPO batching)
        """
        # Get task and store model output
        task = self.session.get_task(task_id)
        if task is None:
            raise ValueError(f"Unknown task_id: {task_id}")
        task.model_output = model_output

        # Dispatch to task-type-specific handler
        if task.task_type == "truthy":
            score, is_correct, task_rewards, aux_score_dict = (
                self._compute_reward_truthy(task)
            )
        else:
            score, is_correct, task_rewards, aux_score_dict = (
                self._compute_reward_standard(task)
            )

        # Accumulate generation and save rewards (cap at num_generations for safety)
        if len(task.generations) < self.num_generations:
            task.add_generation(model_output, task_rewards, score)
            print(
                f"DEBUG: Added generation {len(task.generations)} to task {task_id} (type: {task.task_type})"
            )
        else:
            print(
                f"Warning: Task {task_id} already has {len(task.generations)} generations, not adding more (num_generations={self.num_generations})"
            )
            print(
                f"DEBUG: Task generations: {[g.output[:50] + '...' for g in task.generations]}"
            )

        # Set task correctness
        task.is_correct = is_correct

        return score

    def _compute_reward_truthy(self, task: Task) -> tuple:
        """
        Evaluate a truthy (conversation quality) task.

        Truthy tasks:
        - Primary score defaults to 0.0 (LLM Judge computed in batch via get_rewards())
        - is_correct always False (truthy never affects curriculum)
        - Reward functions: format_think (optional), lang_consistency, repetition, whitespace
        - No format gates applied (format_think is informational only)
        - LLM Judge is deferred to batch processing for efficiency

        Args:
            task: Task object with model_output already stored

        Returns:
            Tuple of (score, is_correct, task_rewards, aux_score_dict)
        """
        if "llm_judge" not in self.aux_reward_functions:
            raise ValueError(
                "Truthy tasks require llm_judge. Please set use_llm_judge=True with api_host, api_port, and model_name"
            )

        # Primary score defaults to 0.0 (LLM Judge computed in batch)
        primary_score = 0.0
        primary_info = "LLM Judge score computed in batch"

        # Build reward list: primary is placeholder (judge computed later)
        primary_reward = RewardFunctionScore(
            score=primary_score,
            reward_function_name="primary",
            info=primary_info,
        )
        task_rewards = [primary_reward]
        aux_score_dict = {}

        # Compute auxiliary rewards (format_think, lang consistency, repetition, whitespace)
        # Format does not gate truthy tasks, just informational
        other_aux_scores = self.get_aux_reward_scores(task, is_correct=False)

        # Add all auxiliary rewards without capping (truthy is continuous)
        for name, data in other_aux_scores.items():
            if name != "llm_judge":  # llm_judge is computed in batch
                aux_reward = RewardFunctionScore(
                    score=data["score"],
                    reward_function_name=name,
                    info=data["info"] or "",
                )
                task_rewards.append(aux_reward)
                aux_score_dict[name] = data

        # Truthy tasks never affect curriculum (is_correct always False)
        score = primary_score
        is_correct = False

        return score, is_correct, task_rewards, aux_score_dict

    def _compute_reward_standard(self, task: Task) -> tuple:
        """
        Evaluate a standard (math/puzzle) task.

        Standard tasks (math, puzzle):
        - Primary score is task correctness (1.0 for correct, 0.0 for incorrect)
        - is_correct = primary_score == 1.0 (hard gate)
        - Format gates: BOTH think and answer tags must be valid
        - On success (format_valid AND is_correct): reward auxiliary fully
        - On failure: cap positive auxiliary rewards at 0
        - LLM Judge computed as auxiliary (separate from primary correctness)

        Args:
            task: Task object with model_output already stored

        Returns:
            Tuple of (score, is_correct, task_rewards, aux_score_dict)
        """
        # Get primary task reward
        if task.task_type not in self.reward_functions:
            raise ValueError(f"Unknown task type: {task.task_type}")

        reward_fn = self.reward_functions[task.task_type]
        result = reward_fn.compute_reward(task)
        primary_score = result.score
        primary_info = result.info

        # Check format validity (both think and answer tags)
        format_valid, format_failure_reason = self._check_format_validity(task)

        # Determine primary success (hard gates: format valid AND primary score == 1.0)
        is_correct = format_valid and (primary_score == 1.0)

        # LLM Judge is deferred and will be computed in batch via get_rewards() for efficiency
        # Do not compute it here
        aux_score_dict = {}

        # Build task rewards based on gates
        primary_reward = RewardFunctionScore(
            score=primary_score,
            reward_function_name="primary",
            info=primary_info or "",
        )
        task_rewards = [primary_reward]

        if format_valid and is_correct:
            # Success: both gates passed
            other_aux_scores = self.get_aux_reward_scores(task, is_correct=True)
            for name, data in other_aux_scores.items():
                if name != "llm_judge":
                    aux_reward = RewardFunctionScore(
                        score=data["score"],
                        reward_function_name=name,
                        info=data["info"] or "",
                    )
                    task_rewards.append(aux_reward)
                    aux_score_dict[name] = data
            score = primary_score
        else:
            # Failure: format invalid or answer incorrect
            other_aux_scores = self.get_aux_reward_scores(task, is_correct=False)

            # Add format feedback
            if not format_valid:
                task_rewards.append(
                    RewardFunctionScore(
                        score=0.0,
                        reward_function_name="format",
                        info=f"Format validation failed: {format_failure_reason}",
                    )
                )

            # Cap positive auxiliary rewards (but not llm_judge)
            for name, data in other_aux_scores.items():
                if name != "llm_judge":
                    capped_score = min(0.0, data["score"])
                    capped_info = (
                        data["info"]
                        if data["score"] < 0
                        else "Primary task failed, positive auxiliary reward capped at 0"
                    )
                    aux_reward = RewardFunctionScore(
                        score=capped_score,
                        reward_function_name=name,
                        info=capped_info or "",
                    )
                    task_rewards.append(aux_reward)
                    aux_score_dict[name] = {"score": capped_score, "info": capped_info}

            score = 0.0

        return score, is_correct, task_rewards, aux_score_dict

    def _check_format_validity(self, task: Task) -> tuple:
        """
        Check format validity for standard tasks.

        Both think and answer tags must be valid for format_valid=True.

        Args:
            task: Task with model_output

        Returns:
            Tuple of (format_valid: bool, failure_reason: str)
        """
        if not self.use_format:
            return True, ""

        format_think_valid = True
        format_answer_valid = True
        failure_reason = ""

        if "format_think" in self.aux_reward_functions:
            format_fn = self.aux_reward_functions["format_think"]
            format_result = format_fn.compute_reward(task, is_correct=False)
            if format_result.score < 1.0:
                format_think_valid = False
                failure_reason = format_result.info

        if "format_answer" in self.aux_reward_functions and task.task_type != "truthy":
            format_fn = self.aux_reward_functions["format_answer"]
            format_result = format_fn.compute_reward(task, is_correct=False)
            if format_result.score < 1.0:
                format_answer_valid = False
                if not failure_reason:
                    failure_reason = format_result.info
        else:
            # Truthy tasks do not require answer tag format validation
            format_answer_valid = True

        return format_think_valid and format_answer_valid, failure_reason

    def _log_completed_task(self, task: Task) -> None:
        """Log a completed task batch to the log file.

        Args:
            task: The completed task
        """
        if self.log_file is None:
            return

        log_entry = task.to_dict()
        log_entry["timestamp"] = datetime.datetime.now().isoformat()

        # Get primary score and info from latest generation
        latest_gen = task.latest_generation
        if latest_gen:
            primary_score = latest_gen.primary_score
            # Find primary reward info
            primary_info = ""
            for reward in latest_gen.rewards:
                if reward.reward_function_name == "primary":
                    primary_info = reward.info or ""
                    break

            # For truthy tasks, primary score IS the judge score
            if task.task_type == "truthy":
                for reward in latest_gen.rewards:
                    if reward.reward_function_name == "llm_judge":
                        primary_score = reward.score
                        break

            # Build aux_scores and info from latest generation rewards (excluding primary and format)
            aux_scores = {}
            aux_info = {}
            for reward in latest_gen.rewards:
                if reward.reward_function_name not in ["primary", "format"]:
                    aux_scores[reward.reward_function_name] = reward.score
                    aux_info[f"aux_{reward.reward_function_name}"] = reward.info or ""
        else:
            primary_score = 0.0
            primary_info = "No generations"
            aux_scores = {}
            aux_info = {}

        log_entry["primary_score"] = primary_score
        log_entry["aux_scores"] = aux_scores
        log_entry["combined_score"] = getattr(task, "combined_score", None)
        log_entry["info"] = {
            "primary": primary_info,
            **aux_info,
        }

        # All GRPO batch information is already included in the generations array
        with open(self.log_file, "a", encoding="utf-8") as f:
            json.dump(log_entry, f, ensure_ascii=False)
            f.write("\n")

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
                self.logged_tasks.clear()
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

        # More conservative success criteria for curriculum learning:
        # Require BOTH: mean >= 0.8 AND at least one perfect response
        # This ensures consistent performance across the batch, not just lucky guesses
        group_success = 1 if (mean_primary >= 0.8 and max_primary == 1.0) else 0

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

    def get_judge_scores(self) -> Dict[str, Any]:
        """Get statistics on LLM Judge scores across all task types.

        Computes average, min, max, and count of LLM Judge scores for all tasks
        that have been evaluated with LLM Judge. This includes both truthy tasks
        (where judge score is primary) and math/puzzle tasks (where judge score is auxiliary).

        Returns:
            Dictionary with:
            - avg_judge_score: Average LLM Judge score (0.0-1.0)
            - min_judge_score: Minimum LLM Judge score
            - max_judge_score: Maximum LLM Judge score
            - judge_score_count: Number of tasks with LLM Judge scores
            - total_tasks_with_judge: Total tasks that have LLM Judge evaluation
        """
        # Compute any missing LLM Judge scores before collecting statistics
        task_ids_to_compute = []
        for task in self.session.tasks.values():
            if not task.latest_generation:
                continue
            has_judge = False
            for reward in task.latest_generation.rewards:
                if reward.reward_function_name == "llm_judge" or (
                    task.task_type == "truthy"
                    and reward.reward_function_name == "primary"
                ):
                    has_judge = True
                    break
            if not has_judge:
                task_ids_to_compute.append(task.task_id)
        if task_ids_to_compute:
            self._compute_batch_llm_judge(task_ids_to_compute)

        judge_scores = []
        tasks_with_judge = 0

        for task in self.session.tasks.values():
            # Skip tasks that haven't been processed yet (no generations)
            if not task.latest_generation:
                continue

            # Extract LLM Judge score from latest generation rewards
            for reward in task.latest_generation.rewards:
                if reward.reward_function_name == "llm_judge":
                    # For math/puzzle tasks, judge score is stored as "llm_judge"
                    judge_scores.append(reward.score)
                    tasks_with_judge += 1
                    break
                elif (
                    task.task_type == "truthy"
                    and reward.reward_function_name == "primary"
                ):
                    # For truthy tasks, primary score IS the judge score
                    judge_scores.append(reward.score)
                    tasks_with_judge += 1
                    break

        if judge_scores:
            avg_judge_score = sum(judge_scores) / len(judge_scores)
            min_judge_score = min(judge_scores)
            max_judge_score = max(judge_scores)
        else:
            avg_judge_score = 0.0
            min_judge_score = 0.0
            max_judge_score = 0.0

        return {
            "avg_judge_score": avg_judge_score,
            "min_judge_score": min_judge_score,
            "max_judge_score": max_judge_score,
            "judge_score_count": len(judge_scores),
            "total_tasks_with_judge": tasks_with_judge,
        }

    def get_prompt(self) -> Optional[Task]:
        """
        Get a task prompt appropriate for current difficulty level.

        With probability reflective_learning_rate (default 0.2), returns a format-failure
        task wrapped in a reflective prompt to help the model learn proper formatting.
        With probability truthy_learning_rate (default 0.2), returns a truthy task for conversation quality evaluation.
        Otherwise, returns a new task based on current difficulty level.

        During warmup stage (first warmup_step iterations), only level 0 tasks are used
        (unless reflective learning or truthy is triggered).

        For GRPO batching: Each call creates a new task instance. The dataset ensures
        that within a batch, the same task is reused for multiple generations.

        Returns:
            Task object with prompt, expected output, etc., or None if no tasks available.
        """
        # Check for reflective learning trigger (only if not disabled and not in warmup)
        if (
            self.reflective_learning_rate > 0
            and random.random() > (1.0 - self.reflective_learning_rate)
            and not self.is_warmup()
        ):
            reflective_task = self.session.get_reflective_task()
            if reflective_task is not None:
                return reflective_task

        # Check for truthy task trigger (20% chance)
        if random.random() < self.truthy_learning_rate and self.session.truthy_tasks:
            selected_task = random.choice(self.session.truthy_tasks)
            return self.session.create_truthy_task(selected_task)

        # Normal task selection logic
        # Collect tasks from all levels with level-based weighting for diversity
        all_available_tasks = []
        level_weights = []

        # During warmup, only use level 0
        if self.is_warmup():
            available_tasks = self.session.tasks_by_level.get(0, [])
            if available_tasks:
                all_available_tasks.extend(available_tasks)
                level_weights.extend([1.0] * len(available_tasks))
            else:
                # Fallback to any available tasks during warmup if level 0 is empty
                for level in range(0, 7):
                    if self.session.tasks_by_level[level]:
                        available_tasks = self.session.tasks_by_level[level]
                        all_available_tasks.extend(available_tasks)
                        level_weights.extend([1.0] * len(available_tasks))
                        break
        else:
            # Normal curriculum: collect from all levels with preference for current level
            current_level_weight = 2.0  # 2x weight for current level
            other_level_weight = 0.5  # 0.5x weight for other levels

            for level in range(0, 7):
                level_tasks = self.session.tasks_by_level.get(level, [])
                if level_tasks:
                    # Limit tasks per level to prevent performance issues
                    max_tasks_per_level = 50
                    if len(level_tasks) > max_tasks_per_level:
                        level_tasks = random.sample(level_tasks, max_tasks_per_level)

                    all_available_tasks.extend(level_tasks)

                    # Apply level-based weighting
                    level_weight = (
                        current_level_weight
                        if level == self.current_level
                        else other_level_weight
                    )
                    level_weights.extend([level_weight] * len(level_tasks))

        if not all_available_tasks:
            return None

        # Weight against recent tasks to ensure diversity (using dataset IDs)
        recent_task_ids = self.session._get_recent_task_ids()

        final_weights = []
        for i, task in enumerate(all_available_tasks):
            weight = level_weights[i]

            # Reduce weight for recent tasks (even if not excluded)
            if task["id"] in recent_task_ids:
                recency_penalty = recent_task_ids.count(task["id"]) / max(
                    len(recent_task_ids), 1
                )
                weight = max(0.1, weight * (1.0 - recency_penalty))
            final_weights.append(weight)

        # Select task with combined weighting
        selected_task = random.choices(all_available_tasks, weights=final_weights, k=1)[
            0
        ]
        if selected_task["type"] == "math":
            task = self.session.create_math_task(selected_task)
            return task
        elif selected_task["type"] == "puzzle":
            task = self.session.create_puzzle_task(selected_task)
            return task

        return None

    def get_learning_stats(self) -> Dict[str, Any]:
        """Get current learning statistics including sliding window success rates."""
        success_stats = self.get_success_rate()
        return {
            "current_level": self.current_level,
            "task_counters": self._get_task_counters(),
            "failed_tasks_count": len(self._get_failed_tasks()),
            "recent_tasks_count": len(self.session._get_recent_task_ids()),
            "available_tasks_by_level": {
                level: len(tasks)
                for level, tasks in self.session.tasks_by_level.items()
            },
            "truthy_tasks_count": len(self.session.truthy_tasks),
            "aux_reward_functions": list(self.aux_reward_functions.keys()),
            "sliding_window_stats": success_stats,
        }

    def get_aux_reward_scores(
        self,
        task: "Task",
        is_correct: bool = False,
    ) -> Dict[str, Dict[str, Any]]:
        """
        Compute all auxiliary reward scores for the given model output.

        Args:
            task: The Task object (contains expected_answer, language, and model_output)
            is_correct: Whether the primary task was answered correctly

        Returns:
            Dictionary mapping auxiliary reward function names to dicts with 'score' and 'info'
        """
        aux_scores = {}
        is_truthy_task = task.task_type == "truthy"

        for aux_name, aux_fn in self.aux_reward_functions.items():
            if is_truthy_task and aux_name == "format_answer":
                # Skip format_answer for truthy tasks
                continue
            if aux_name == "llm_judge":
                # Skip llm_judge as it's computed in batch via get_rewards()
                continue
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

    def _compute_batch_llm_judge(self, task_ids: List[str]) -> None:
        """
        Compute LLM Judge rewards for multiple tasks in batch (efficiently via API).

        This method is called from get_rewards() before scoring to enable efficient
        batched LLM Judge API calls. Each task's primary reward is updated with the
        LLM Judge score (for truthy) or added as auxiliary (for math/puzzle).

        Modifies task.latest_generation.rewards in place to include LLM Judge results.

        Args:
            task_ids: List of task identifiers to evaluate with LLM Judge
        """
        if not self.use_llm_judge or "llm_judge" not in self.aux_reward_functions:
            return  # LLM Judge not enabled

        # Collect all tasks that need LLM Judge evaluation
        tasks_to_judge = []
        task_index_map = {}  # Maps index in tasks_to_judge to task_id

        for task_id in task_ids:
            task = self.session.get_task(task_id)
            if task is None:
                continue
            # Skip tasks that haven't been processed yet (no generations)
            if not task.latest_generation:
                continue
            # Collect all tasks that need LLM Judge evaluation (including truthy)
            # Check if any generation needs judge (not just latest)
            needs_judge = False
            for gen in task.generations:
                has_judge = any(
                    r.reward_function_name == "llm_judge" for r in gen.rewards
                )
                if not has_judge:
                    needs_judge = True
                    break
            if needs_judge:
                task_index_map[len(tasks_to_judge)] = task_id
                tasks_to_judge.append(task)

        if not tasks_to_judge:
            return  # No tasks need LLM Judge evaluation

        # Batch evaluate all tasks with LLM Judge
        try:
            llm_judge_fn = self.aux_reward_functions["llm_judge"]
            # Batch API call (always use batch method)
            judge_results = llm_judge_fn.compute_rewards_batch(tasks_to_judge)

            # Update task rewards with LLM Judge scores
            for idx, judge_scores in enumerate(judge_results):
                task_id = task_index_map[idx]
                task = self.session.get_task(task_id)
                if not task:
                    continue

                if task.task_type == "truthy":
                    # For truthy: replace primary score with LLM Judge score for each generation
                    for gen_idx, gen in enumerate(task.generations):
                        if gen_idx < len(judge_scores):
                            judge_score = judge_scores[gen_idx]
                            for i, reward in enumerate(gen.rewards):
                                if reward.reward_function_name == "primary":
                                    gen.rewards[i] = RewardFunctionScore(
                                        score=judge_score.score,
                                        reward_function_name="primary",
                                        info=judge_score.info or "",
                                    )
                                    gen.primary_score = judge_score.score
                                    break
                else:
                    # For math/puzzle: add as auxiliary reward to each generation that doesn't have it
                    for gen_idx, gen in enumerate(task.generations):
                        if gen_idx < len(judge_scores):
                            judge_score = judge_scores[gen_idx]
                            # Check if this generation already has llm_judge
                            has_judge = any(
                                r.reward_function_name == "llm_judge"
                                for r in gen.rewards
                            )
                            if not has_judge:
                                gen.rewards.append(
                                    RewardFunctionScore(
                                        score=judge_score.score,
                                        reward_function_name="llm_judge",
                                        info=judge_score.info or "",
                                    )
                                )
        except Exception as e:
            print(f"Warning: Batch LLM Judge evaluation failed: {e}")
            # Gracefully handle errors by skipping judge evaluation
            # Tasks will use placeholder scores

    def get_rewards(self, task_ids: List[str]) -> List[List[float]]:
        """Calculate combined reward scores for multiple completed tasks.

        For each task, computes combined scores for each generation in the task.
        Retrieves primary correctness and auxiliary scores from generation.rewards,
        normalizes each auxiliary score to [0, 1] range, then applies aux_weight
        to blend primary with average normalized auxiliary scores.

        Normalization strategy:
        - Primary score: already binary (0 or 1)
        - Auxiliary scores: clip to [-1, 1], then shift to [0, 1]
          (assumes auxiliary rewards designed to be in [-1, 1] range)
        - Combined: primary_weight * primary + aux_weight * avg(normalized_aux) + judge_weight * judge

        Args:
            task_ids: List of task identifiers (should have been processed by compute_reward)

        Returns:
            List of lists of combined reward scores in the range [0, 1] suitable for RL training.
            Each sublist contains scores for each generation in the corresponding task.

        Raises:
            ValueError: If any task_id not found in session
        """
        # Batch process LLM Judge for all tasks (for efficiency)
        self._compute_batch_llm_judge(task_ids)

        # Finalize completed batches: curriculum tracking and logging
        for task_id in task_ids:
            task = self.session.get_task(task_id)
            if task is None:
                continue

            if len(task.generations) >= self.num_generations:
                if len(task.generations) > self.num_generations:
                    raise ValueError(
                        f"Task {task_id} has {len(task.generations)} generations, expected {self.num_generations}"
                    )

                # Only log and track success for tasks we haven't logged before
                if task_id not in self.logged_tasks:
                    self.logged_tasks.add(task_id)

                    # Complete group: track success at prompt level
                    primary_scores = [g.primary_score for g in task.generations]
                    if task.task_type != "truthy":
                        self._track_success_group(task.level, primary_scores)

                    # Increment step counter ONCE per complete prompt group
                    self.global_step += 1

                    # Check if we should advance/demote level
                    self._update_level()

                    # Log evaluation if configured
                    self._log_completed_task(task)

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
                # Skip tasks that haven't been processed yet
                if not task.latest_generation:
                    continue

                for reward in task.latest_generation.rewards:
                    if reward.reward_function_name == "llm_judge":
                        if reward.score > highest_judge_score:
                            highest_judge_score = reward.score
                        if reward.score < lowest_judge_score:
                            lowest_judge_score = reward.score
                    elif (
                        reward.reward_function_name == "primary"
                        and task.task_type == "truthy"
                    ):
                        # For truthy tasks, judge score is stored as primary
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

            gen_combined_scores = []
            for gen in task.generations:
                # Extract primary and auxiliary scores from generation.rewards
                primary_score = None
                judge_score = 0.0
                aux_scores = {}
                is_truthy_task = task.task_type == "truthy"

                for reward in gen.rewards:
                    if reward.reward_function_name == "primary":
                        primary_score = reward.score
                    elif reward.reward_function_name == "llm_judge":
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

                # For truthy tasks, the judge score is stored as primary score
                if is_truthy_task:
                    judge_score = primary_score
                    # Normalize the judge score for truthy tasks
                    judge_score = self._normalize_score(
                        judge_score, lowest_judge_score, highest_judge_score
                    )
                    primary_score = judge_score

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
                gen_combined_scores.append(combined_score)

            combined_rewards.append(gen_combined_scores)

            # Store the final combined score on the task for validation (use last generation)
            task.combined_score = gen_combined_scores[-1] if gen_combined_scores else 0

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
