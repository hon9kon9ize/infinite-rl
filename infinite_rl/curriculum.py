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


class CurriculumLearning:

    def __init__(
        self,
        session: Optional[Session] = None,
        timeout: int = 10,
        answer_tag: str = "answer",
        think_tag: str = "think",
        aux_weight: float = 0.2,
        llm_judge_weight: float = 0.2,
        use_lang_consistency: bool = True,
        use_format: bool = True,
        use_reasoning_steps: bool = True,
        use_length: bool = False,
        use_llm_judge: bool = False,
        reasoning_language: str = "en",
        lang_consistency_kwargs: Optional[Dict[str, Any]] = None,
        format_kwargs: Optional[Dict[str, Any]] = None,
        reasoning_steps_kwargs: Optional[Dict[str, Any]] = None,
        length_kwargs: Optional[Dict[str, Any]] = None,
        llm_judge_kwargs: Optional[Dict[str, Any]] = None,
        log_file: Optional[str] = None,
        window_size: int = 50,
        success_rate_threshold: float = 0.7,
        variance_threshold: float = 0.15,
        demote_threshold: float = 0.4,
        warmup_step: int = 16,
        truthy_learning_rate: float = 0.1,
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
            aux_weight: Weight for auxiliary rewards in combined score (0-1, default: 0.2)
            llm_judge_weight: Weight for LLM Judge reward, computed independently of format/correctness gates (0-1, default: 0.2)
            use_lang_consistency: Enable language consistency auxiliary reward (default: True)
            use_format: Enable format validation auxiliary reward (default: True)
            use_reasoning_steps: Enable chain-of-thought reasoning steps bonus (default: True)
            use_length: Enable response length auxiliary reward (default: False)
            use_llm_judge: Enable LLM-based quality evaluation via remote sglang server (default: False)
            reasoning_language: ISO language code for reasoning analysis (default: "en")
            lang_consistency_kwargs: Keyword arguments for LangConsistencyRewardFunction
            format_kwargs: Keyword arguments for FormatRewardFunction
            reasoning_steps_kwargs: Keyword arguments for ReasoningStepsRewardFunction
            length_kwargs: Keyword arguments for LengthRewardFunction
            llm_judge_kwargs: Keyword arguments for LLMJudgeRewardFunction (api_host, api_port, model_name, etc.)
            log_file: Path to the logging file (JSON Lines format). If None, defaults to 'curriculum_log.jsonl' in the module directory.
            window_size: Size of the sliding window for success rate tracking (default: 50)
            success_rate_threshold: Required success rate for difficulty increase (default: 0.7 = 70%)
            variance_threshold: Maximum variance for success rate stability (default: 0.15)
            demote_threshold: Success rate threshold for difficulty decrease (default: 0.4 = 40%)
            warmup_step: Number of initial steps to only use level 0 tasks (default: 32)

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
        self.use_format = use_format
        self.use_reasoning_steps = use_reasoning_steps
        self.use_length = use_length
        self.use_llm_judge = use_llm_judge
        self.lang_consistency_kwargs = lang_consistency_kwargs or {}
        self.format_kwargs = format_kwargs or {}
        self.reasoning_steps_kwargs = reasoning_steps_kwargs or {}
        self.length_kwargs = length_kwargs or {}
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
                    task_name="length",
                    timeout=self.timeout,
                    answer_tag=self.answer_tag,
                    think_tag=self.think_tag,
                    **self.length_kwargs,
                )
            except Exception as e:
                print(f"Warning: Could not initialize LengthRewardFunction: {e}")

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

    def compute_reward(
        self,
        task_id: str,
        model_output: str,
    ) -> float:
        """
        Evaluate a single model output and update learning state.

        Dispatches to task-type-specific handlers:
        - Truthy tasks: LLM Judge as primary score
        - Math/Puzzle: Primary task correctness with format gates

        Args:
            task_id: Task identifier
            model_output: Raw model response (string)

        Returns:
            Reward score (float)
        """
        # Single completion path
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
        else:
            print(
                f"Warning: Task {task_id} already has {len(task.generations)} generations, not adding more (num_generations={self.num_generations})"
            )
            print(
                f"DEBUG: Task generations: {[g.output[:50] + '...' for g in task.generations]}"
            )

        # Finalize batch if complete
        combined_score = score  # Default to primary score for incomplete batches
        if len(task.generations) >= self.num_generations:
            scores = self._finalize_batch([task_id])
            combined_score = scores[0] if scores else score

        # Set task correctness
        task.is_correct = is_correct

        return combined_score

    def compute_rewards(self, task_id: str, model_outputs: List[str]) -> List[float]:
        """
        Evaluate multiple completions for a task and compute rewards in batch.

        Accumulates all generations first, then finalizes batch once:
        - Computes LLM Judge scores
        - Computes combined scores for all generations
        - Tracks curriculum success
        - Updates difficulty level
        - Logs completed task

        Args:
            task_id: Task identifier
            model_outputs: List of model responses

        Returns:
            List of reward scores corresponding to each completion
        """
        # Accumulate all generations (don't finalize yet)
        for completion in model_outputs:
            task = self.session.get_task(task_id)
            if task is None:
                raise ValueError(f"Unknown task_id: {task_id}")
            task.model_output = completion

            # Dispatch to task-type-specific handler
            if task.task_type == "truthy":
                score, is_correct, task_rewards, aux_score_dict = (
                    self._compute_reward_truthy(task)
                )
            else:
                score, is_correct, task_rewards, aux_score_dict = (
                    self._compute_reward_standard(task)
                )

            # Accumulate generation without finalization
            if len(task.generations) < self.num_generations:
                task.add_generation(completion, task_rewards, score)
            else:
                print(
                    f"Warning: Task {task_id} already has {len(task.generations)} generations, not adding more (num_generations={self.num_generations})"
                )

            task.is_correct = is_correct

        # After all generations accumulated, finalize batch once
        return self._finalize_batch([task_id])

    def _finalize_batch(self, task_ids: List[str]) -> List[float]:
        """
        Finalize batch processing: compute LLM Judge, combined scores, and curriculum tracking.

        This is called after all generations for a task are accumulated.

        Args:
            task_ids: List of task IDs to finalize (typically single-element list)

        Returns:
            List of combined scores for all generations of the first task
        """
        scores = []

        for task_id in task_ids:
            task = self.session.get_task(task_id)
            if not task or len(task.generations) < self.num_generations:
                # Not ready for finalization yet
                for gen in task.generations if task else []:
                    scores.append(gen.primary_score)
                continue

            if task_id in self.logged_tasks:
                # Already finalized
                for gen in task.generations:
                    scores.append(
                        gen.combined_score
                        if gen.combined_score is not None
                        else gen.primary_score
                    )
                continue

            self.logged_tasks.add(task_id)

            # Step 1: Compute batch LLM Judge (updates generation rewards in place)
            self._compute_batch_llm_judge([task_id])

            # Step 2: Recompute combined scores for all generations (now with judge scores available)
            # This ensures combined_score is ALWAYS set, even if LLM Judge fails
            for gen in task.generations:
                if (
                    gen.combined_score is None
                ):  # Only compute if not already set by LLM Judge
                    gen.combined_score = self._compute_combined_score(task, gen)

            # Safety check: ensure all generations have combined_score set
            for gen in task.generations:
                if gen.combined_score is None:
                    # Fallback to primary score if combined score computation failed
                    gen.combined_score = gen.primary_score

            # Collect scores to return
            for gen in task.generations:
                scores.append(gen.combined_score)

            # Step 3: Track success at prompt level for curriculum
            primary_scores = [g.primary_score for g in task.generations]
            if task.task_type != "truthy":
                self._track_success_group(task.level, primary_scores)

            # Increment step counter ONCE per complete prompt group
            self.global_step += 1

            # Check if we should advance/demote level
            self._update_level()

            # Log evaluation if configured
            self._log_completed_task(task)

        return scores

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
        - is_correct is True only if BOTH format is valid AND primary score is 1.0
        - Format gates: BOTH think and answer tags must be valid
        - On success (format_valid AND is_correct): reward auxiliary fully
        - On failure: include auxiliary rewards as-is
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

        # LLM Judge is deferred and will be computed in batch via get_rewards() for efficiency
        # Do not compute it here
        aux_score_dict = {}

        # Build task rewards based on primary score
        primary_reward = RewardFunctionScore(
            score=primary_score,
            reward_function_name="primary",
            info=primary_info or "",
        )
        task_rewards = [primary_reward]

        if primary_score == 1.0:
            # Success: primary score is correct
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
            # Failure: primary score incorrect
            other_aux_scores = self.get_aux_reward_scores(task, is_correct=False)

            # Include auxiliary rewards as-is (no capping)
            for name, data in other_aux_scores.items():
                if name != "llm_judge":
                    aux_reward = RewardFunctionScore(
                        score=data["score"],
                        reward_function_name=name,
                        info=data["info"] or "",
                    )
                    task_rewards.append(aux_reward)
                    aux_score_dict[name] = data

            score = 0.0

        # Check format validity from rewards (already computed in get_aux_reward_scores)
        # Both format_think and format_answer must be valid (score >= 1.0)
        format_valid = True
        if self.use_format:
            # Check if format rewards exist and are valid
            has_format_think = False
            has_format_answer = False

            for reward in task_rewards:
                if reward.reward_function_name == "format_think":
                    if reward.score < 1.0:
                        format_valid = False
                    has_format_think = True
                elif reward.reward_function_name == "format_answer":
                    if reward.score < 1.0:
                        format_valid = False
                    has_format_answer = True

            # For non-truthy tasks, both format rewards are required
            if task.task_type != "truthy":
                if not has_format_think or not has_format_answer:
                    # If format rewards not computed, fall back to checking
                    # This shouldn't happen if use_format=True
                    format_valid = False

        # Apply format gate: if format invalid, set score to 0.0
        if not format_valid and score > 0.0:
            score = 0.0

        # Determine is_correct: True only if format valid AND primary score is 1.0
        if primary_score == 1.0 and format_valid:
            is_correct = True
        else:
            is_correct = False

        return score, is_correct, task_rewards, aux_score_dict

    def _check_format_validity(self, task: Task, generation: "Generation") -> tuple:
        """
        Check format validity for a specific generation.

        Both think and answer tags must be valid for format_valid=True.

        Args:
            task: Task object (used only for task_type and auxiliary functions)
            generation: Generation to check format for. Uses generation.output.

        Returns:
            Tuple of (format_valid: bool, failure_reason: str)
        """
        if not self.use_format:
            return True, ""

        # Temporarily swap model_output to check format of specific generation
        original_output = task.model_output
        task.model_output = generation.output

        try:
            format_think_valid = True
            format_answer_valid = True
            failure_reason = ""

            if "format_think" in self.aux_reward_functions:
                format_fn = self.aux_reward_functions["format_think"]
                format_result = format_fn.compute_reward(task, is_correct=False)
                if format_result.score < 1.0:
                    format_think_valid = False
                    failure_reason = format_result.info

            if (
                "format_answer" in self.aux_reward_functions
                and task.task_type != "truthy"
            ):
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
        finally:
            # Restore original output
            task.model_output = original_output

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

        # All GRPO batch information is already included in the generations array
        # Task-level logging only includes task metadata; generation-level info is in generations[]
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

        # Success criteria: count as success if ANY response is perfect (1.0)
        # This allows curriculum progression when model occasionally gets correct answers
        group_success = 1 if max_primary == 1.0 else 0

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

        With probability truthy_learning_rate (default 0.2), returns a truthy task for conversation quality evaluation.
        Otherwise, returns a new task based on current difficulty level.

        During warmup stage (first warmup_step iterations), only level 0 tasks are used
        (unless truthy is triggered).

        For GRPO batching: Each call creates a new task instance. The dataset ensures
        that within a batch, the same task is reused for multiple generations.

        Returns:
            Task object with prompt, expected output, etc., or None if no tasks available.
        """
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
            # Normal curriculum: sample from level 0 through current level for diversity
            # This allows the model to see a mix of difficulties while progressing
            for level in range(0, self.current_level + 1):
                level_tasks = self.session.tasks_by_level.get(level, [])
                if level_tasks:
                    all_available_tasks.extend(level_tasks)
                    # Weight current level tasks higher (10x) to focus training
                    weight = 10.0 if level == self.current_level else 1.0
                    level_weights.extend([weight] * len(level_tasks))

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

        Format gate applies to judge: if format is invalid, final reward is zero.

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
                    # For truthy: replace primary score with LLM Judge score
                    # Apply format gate: if format invalid, gate judge reward to 0.0
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
                                    # Recompute combined_score after updating judge score
                                    gen.combined_score = self._compute_combined_score(
                                        task, gen
                                    )
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
                                # Recompute combined_score after adding judge reward
                                gen.combined_score = self._compute_combined_score(
                                    task, gen
                                )
        except Exception as e:
            print(f"Warning: Batch LLM Judge evaluation failed: {e}")
            # Gracefully handle errors by skipping judge evaluation
            # Tasks will use placeholder scores

    def _compute_combined_score(self, task: Task, generation: "Generation") -> float:
        """Compute combined score for a generation.

        Format gate applies: if format is invalid, combined score is gated to 0.0.
        For math/puzzle tasks, correctness gating applies: if generation is incorrect,
        length and llm_judge scores are set to 0.0, and their info fields are updated
        to indicate gating.
        Format validity is extracted from generation rewards (format_think, format_answer).

        Args:
            task: The task
            generation: The generation

        Returns:
            Combined score in [0, 1]
        """
        primary_score = generation.primary_score
        aux_scores = {}
        judge_score = 0.0
        for reward in generation.rewards:
            if reward.reward_function_name == "llm_judge":
                judge_score = reward.score
            elif reward.reward_function_name not in [
                "primary",
            ]:
                aux_scores[reward.reward_function_name] = reward.score

        # Apply correctness gating for math and puzzle tasks
        if task.task_type in ["math", "puzzle"] and not generation.is_correct:
            if "length" in aux_scores:
                aux_scores["length"] = 0.0
                # Update info for length reward
                for reward in generation.rewards:
                    if reward.reward_function_name == "length":
                        reward.info = f"Gated due to incorrect generation: {reward.info or ''}"
                        break
            judge_score = 0.0
            # Update info for llm_judge reward
            for reward in generation.rewards:
                if reward.reward_function_name == "llm_judge":
                    reward.info = f"Gated due to incorrect generation: {reward.info or ''}"
                    break

        # Normalize aux (assuming aux scores are now in [0, 1])
        if aux_scores:
            aux_avg = sum(aux_scores.values()) / len(aux_scores)
        else:
            aux_avg = 0

        # Weights
        if task.task_type == "truthy":
            primary_weight = 1.0 - self.aux_weight
        else:
            if self.use_llm_judge:
                primary_weight = 1.0 - self.aux_weight - self.llm_judge_weight
            else:
                primary_weight = 1.0 - self.aux_weight

        judge_contribution = (
            self.llm_judge_weight * judge_score
            if (self.use_llm_judge and task.task_type != "truthy")
            else 0.0
        )
        combined_score = max(
            0.0,
            min(
                1.0,
                primary_weight * primary_score
                + self.aux_weight * aux_avg
                + judge_contribution,
            ),
        )
        return combined_score

    def is_warmup(self) -> bool:
        """Check if currently in warmup stage."""
        return self.global_step < self.warmup_step
