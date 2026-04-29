"""
Simplified training script for fine-tuning Gemma-3-4B with TRL's GRPO trainer
using Infinite-RL's curriculum learning and reward functions.

This version is optimized for vLLM COLOCATE mode ONLY (simplified, no server mode).

NOTE: This script assumes the model vocabulary has already been expanded using train/model_expand.py
and special tokens have been added. It does NOT handle vocabulary expansion.

Integrates:
- TRL's GRPO (Group Relative Policy Optimization) trainer
- vLLM in colocate mode for efficient generation
- Infinite-RL's curriculum learning for progressive difficulty
- Custom reward functions from Infinite-RL for task evaluation
"""

# import unsloth
import json
import argparse
from pathlib import Path
from typing import Optional, List, Callable
from dataclasses import dataclass, asdict

import torch
import numpy as np
import wandb
from transformers import (
    Gemma3ForCausalLM,
    AutoTokenizer,
    TrainerCallback,
)
from trl import GRPOTrainer, GRPOConfig
from peft import LoraConfig, get_peft_model, TaskType

# Import infinite-rl components
from infinite_rl.curriculum import CurriculumLearning
from infinite_rl.dynamic_dataset import DynamicCurriculumDataset
import functools


def make_json_serializable(obj):
    """Convert non-JSON-serializable objects to strings recursively.

    Handles: torch.dtype, numpy.dtype, Path, and other common types.
    """
    if isinstance(obj, dict):
        return {k: make_json_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [make_json_serializable(item) for item in obj]
    elif isinstance(obj, (torch.dtype, np.dtype)):
        return str(obj)
    elif isinstance(obj, Path):
        return str(obj)
    elif isinstance(obj, type):
        return obj.__name__
    elif hasattr(obj, "__dict__") and not isinstance(
        obj, (str, int, float, bool, type(None))
    ):
        # For custom objects, try to convert to dict
        try:
            return make_json_serializable(obj.__dict__)
        except:
            return str(obj)
    else:
        return obj


@dataclass
class InfiniteRLConfig:
    """Configuration for Infinite-RL curriculum integration."""

    # Curriculum parameters
    window_size: int = 80
    success_rate_threshold: float = 0.65
    variance_threshold: float = 0.15
    demote_threshold: float = 0.4
    warmup_step: int = 100
    level_change_cooldown: int = 10

    num_generations: int = 4

    # Reward function parameters
    timeout: int = 40
    answer_tag: str = "answer"
    think_tag: str = "think"
    reasoning_language: str = "en"
    reasoning_template: bool = False

    # Auxiliary rewards
    use_format: bool = True
    use_reasoning_steps: bool = True
    use_lang_consistency: bool = True
    use_length: bool = True
    aux_weight: float = 0.2

    # LLM Judge configuration
    use_llm_judge: bool = False
    llm_judge_host: str = "localhost"
    llm_judge_port: int = 8000
    llm_judge_model: str = "Skywork/Skywork-Reward-V2-Qwen3-4B"
    llm_judge_weight: float = 0.1

    # Output
    log_file: Optional[str] = "curriculum_learning_log.jsonl"


def create_curriculum(config: InfiniteRLConfig) -> CurriculumLearning:
    """Initialize the curriculum learning system from Infinite-RL.

    Args:
        config: Infinite-RL configuration

    Returns:
        Initialized CurriculumLearning instance
    """
    # Prepare LLM Judge kwargs if enabled
    llm_judge_kwargs = {}
    if config.use_llm_judge:
        llm_judge_kwargs = {
            "api_host": config.llm_judge_host,
            "api_port": config.llm_judge_port,
            "model_name": config.llm_judge_model,
        }

    curriculum = CurriculumLearning(
        timeout=config.timeout,
        answer_tag=config.answer_tag,
        think_tag=config.think_tag,
        reasoning_language=config.reasoning_language,
        reasoning_template=config.reasoning_template,
        aux_weight=config.aux_weight,
        use_format=config.use_format,
        use_reasoning_steps=config.use_reasoning_steps,
        use_lang_consistency=config.use_lang_consistency,
        use_llm_judge=config.use_llm_judge,
        use_length=config.use_length,
        llm_judge_kwargs=llm_judge_kwargs,
        window_size=config.window_size,
        llm_judge_weight=config.llm_judge_weight,
        success_rate_threshold=config.success_rate_threshold,
        variance_threshold=config.variance_threshold,
        demote_threshold=config.demote_threshold,
        warmup_step=config.warmup_step,
        level_change_cooldown=config.level_change_cooldown,
        num_generations=config.num_generations,
        log_file=config.log_file,
        truthy_learning_rate=0,
    )
    return curriculum


def build_dynamic_dataset(
    curriculum: CurriculumLearning,
    num_samples: int = 10000,
) -> DynamicCurriculumDataset:
    """Build a dynamic dataset that generates prompts on-demand.

    This ensures each sample always reflects the current curriculum level
    without needing to pre-generate or refresh the dataset.

    Args:
        curriculum: Initialized CurriculumLearning instance
        num_samples: Virtual size of the dataset (for trainer iteration)

    Returns:
        DynamicCurriculumDataset instance
    """
    print(f"\n[DEBUG] Creating dynamic dataset:")
    print(f"  - Virtual size: {num_samples} samples")
    print(f"  - Prompts generated on-demand from curriculum")
    print(
        f"  - Current curriculum level: {curriculum.get_learning_stats()['current_level']}"
    )

    return DynamicCurriculumDataset(curriculum, num_samples)


def create_curriculum_reward_func(
    curriculum: CurriculumLearning,
) -> Callable:
    """Create a reward function compatible with GRPOTrainer.

    The reward function uses curriculum.compute_reward() which:
    1. Computes primary + auxiliary rewards
    2. Accumulates generations
    3. When batch complete: computes LLM Judge, recomputes combined scores
    4. Returns the combined score directly

    Args:
        curriculum: Initialized CurriculumLearning instance

    Returns:
        A callable that computes rewards for completions
    """

    def reward_func(
        completions: List[str],
        **kwargs,
    ) -> List[float]:
        """Compute rewards for completions using Infinite-RL's curriculum.

        Args:
            completions: List of model completions
            **kwargs: Additional arguments from trainer, including task_metadata

        Returns:
            List of reward scores in range [0, 1] (combined scores with judge)
        """
        task_metadata = kwargs.get("task_metadata")
        from collections import defaultdict

        # CRITICAL: Preserve original completion order by storing (index, completion) pairs
        # This prevents rewards from being assigned to wrong generations
        indexed_completions = []  # List of (original_index, task_id, completion)
        grouped = defaultdict(list)  # task_id -> list of (original_index, completion)

        for i, completion in enumerate(completions):
            if task_metadata and i < len(task_metadata):
                task_id = task_metadata[i]["task_id"]
            else:
                # Fallback if no metadata
                task_id = f"task_{i // curriculum.num_generations}"

            indexed_completions.append((i, task_id, completion))
            grouped[task_id].append((i, completion))

        # Collect all rewards with their original indices
        reward_with_index = {}  # original_index -> reward_score

        # Process each task's completions in batch
        for task_id, index_completion_list in grouped.items():
            try:
                # Extract content from TRL's chat format for all completions
                completion_texts = []
                indices = []
                for orig_idx, completion in index_completion_list:
                    indices.append(orig_idx)
                    completion_text = completion
                    if isinstance(completion, list) and len(completion) > 0:
                        # Extract from list of dicts format
                        completion_text = completion[0].get("content", "")
                    elif isinstance(completion, dict):
                        # Extract from single dict format
                        completion_text = completion.get("content", "")

                    # Ensure we have a string
                    if not isinstance(completion_text, str):
                        print(
                            f"Warning: Completion for task {task_id} is not a string after extraction. Type: {type(completion_text)}, value: {completion_text}"
                        )
                        completion_texts.append("")
                    else:
                        completion_texts.append(completion_text)

                # Compute rewards for all completions in batch
                # Returns combined_scores (includes LLM Judge if batch complete)
                batch_scores = curriculum.compute_rewards(task_id, completion_texts)

                # Map scores back to original indices
                for idx, score in zip(indices, batch_scores):
                    reward_with_index[idx] = float(score)

            except Exception as e:
                print(f"Warning: Error computing reward for task {task_id}: {e}")
                # Add zero scores for all completions in this task
                for idx, _ in index_completion_list:
                    reward_with_index[idx] = 0.0

        # Reconstruct rewards in original order
        rewards_list = [reward_with_index.get(i, 0.0) for i in range(len(completions))]
        return rewards_list

    return reward_func


class CurriculumLoggingCallback(TrainerCallback):
    """Callback to log curriculum statistics during training.

    No dataset refresh needed - the dynamic dataset automatically reflects
    the current curriculum level.
    """

    def __init__(self, curriculum: CurriculumLearning, log_every_n_steps: int = 10):
        """Initialize the callback.

        Args:
            curriculum: The CurriculumLearning instance to monitor
            log_every_n_steps: Log curriculum stats every N steps
        """
        self.curriculum = curriculum
        self.log_every_n_steps = log_every_n_steps
        self.last_log_step = 0

    def on_step_end(self, args, state, control, **kwargs):
        """Log curriculum statistics periodically."""
        if state.global_step - self.last_log_step >= self.log_every_n_steps:
            stats = self.curriculum.get_learning_stats()
            current_level = stats.get("current_level", 0)

            # Log to wandb
            if wandb.run is not None:
                wandb_logs = {
                    "curriculum/current_level": current_level,
                    "curriculum/global_step": self.curriculum.global_step,
                }

                # Log success rates by level
                sliding_window = stats.get("sliding_window_stats", {})

                # Log aggregated stats (these are always at top level)
                if "mean_success_rate" in sliding_window:
                    wandb_logs["curriculum/mean_success_rate"] = sliding_window.get(
                        "mean_success_rate", 0
                    )
                    wandb_logs["curriculum/mean_variance"] = sliding_window.get(
                        "mean_variance", 0
                    )
                    wandb_logs["curriculum/total_samples"] = sliding_window.get(
                        "samples", 0
                    )

                # Log per-level stats
                for level_stat in sliding_window.get("by_level", []):
                    level = level_stat.get("level", "unknown")
                    success_rate = level_stat.get("success_rate", 0)
                    variance = level_stat.get("variance", 0)
                    samples = level_stat.get("samples", 0)

                    wandb_logs[f"curriculum/level_{level}/success_rate"] = success_rate
                    wandb_logs[f"curriculum/level_{level}/variance"] = variance
                    wandb_logs[f"curriculum/level_{level}/samples"] = samples

                # Log judge scores (if LLM Judge enabled)
                if self.curriculum.use_llm_judge:
                    truthy_stats = self.curriculum.get_judge_scores()
                    wandb_logs["curriculum/judge/avg_judge_score"] = truthy_stats.get(
                        "avg_judge_score", 0.0
                    )
                    wandb_logs["curriculum/judge/min_judge_score"] = truthy_stats.get(
                        "min_judge_score", 0.0
                    )
                    wandb_logs["curriculum/judge/max_judge_score"] = truthy_stats.get(
                        "max_judge_score", 0.0
                    )
                    wandb_logs["curriculum/judge/count"] = truthy_stats.get(
                        "judge_score_count", 0
                    )

                # Log generation statistics for recent tasks (Phase 4: Enhanced generation tracking)
                recent_tasks = list(self.curriculum.session.tasks.keys())[
                    -10:
                ]  # Last 10 tasks
                total_generations = 0
                total_correct_generations = 0
                generation_scores = []

                for task_id in recent_tasks:
                    batch_stats = self.curriculum.session.get_batch_stats(task_id)
                    if batch_stats:
                        total_generations += batch_stats["num_generations"]
                        total_correct_generations += batch_stats["correct_generations"]
                        generation_scores.extend(
                            [batch_stats["scores"]["avg"]]
                            if batch_stats["num_generations"] > 0
                            else []
                        )

                        # Log per-task generation stats (sample a few)
                        if (
                            len(recent_tasks) <= 3
                        ):  # Only log detailed stats for very recent tasks
                            wandb_logs[
                                f"generations/task_{task_id}/num_generations"
                            ] = batch_stats["num_generations"]
                            wandb_logs[f"generations/task_{task_id}/avg_score"] = (
                                batch_stats["scores"]["avg"]
                            )
                            wandb_logs[f"generations/task_{task_id}/best_score"] = (
                                batch_stats["scores"]["max"]
                            )
                            wandb_logs[f"generations/task_{task_id}/correct_ratio"] = (
                                batch_stats["correct_generations"]
                                / batch_stats["num_generations"]
                                if batch_stats["num_generations"] > 0
                                else 0
                            )

                # Log aggregated generation statistics
                if total_generations > 0:
                    wandb_logs["generations/recent_tasks_count"] = len(recent_tasks)
                    wandb_logs["generations/total_generations"] = total_generations
                    wandb_logs["generations/avg_correct_ratio"] = (
                        total_correct_generations / total_generations
                    )
                    if generation_scores:
                        wandb_logs["generations/avg_task_score"] = sum(
                            generation_scores
                        ) / len(generation_scores)

                wandb.log(wandb_logs)

            self.last_log_step = state.global_step


def setup_training_args(
    output_dir: str = "./grpo-gemma3-4b",
    learning_rate: float = 1e-5,
    num_train_epochs: int = 3,
    per_device_train_batch_size: int = 4,
    num_generations: int = 4,
    max_completion_length: int = 512,
    max_prompt_length: int = 2048,
    gradient_accumulation_steps: int = 1,
    warmup_steps: int = 0,
    vllm_mode: str = "colocate",
    vllm_server_base_url: Optional[str] = None,
    **kwargs,
) -> GRPOConfig:
    """Setup GRPO training configuration for vLLM.

    Args:
        output_dir: Output directory for checkpoints
        learning_rate: Learning rate for training
        num_train_epochs: Number of training epochs
        per_device_train_batch_size: Batch size per device
        num_generations: Number of completions per prompt
        max_completion_length: Maximum completion length
        max_prompt_length: Maximum prompt length (for vLLM context size)
        gradient_accumulation_steps: Number of steps to accumulate gradients
        warmup_steps: Number of warmup steps for learning rate scheduling
        vllm_mode: vLLM mode ('server' or 'colocate')
        vllm_server_base_url: Base URL for vLLM server mode
        **kwargs: Additional arguments to pass to GRPOConfig

    Returns:
        Configured GRPOConfig instance
    """
    config = GRPOConfig(
        output_dir=output_dir,
        learning_rate=learning_rate,
        num_train_epochs=num_train_epochs,
        per_device_train_batch_size=per_device_train_batch_size,
        num_generations=num_generations,
        max_completion_length=max_completion_length,
        # max_prompt_length=max_prompt_length, # TRL GRPOConfig not supports max_prompt_length
        gradient_accumulation_steps=gradient_accumulation_steps,
        warmup_steps=warmup_steps,
        temperature=0.9,
        top_p=0.9,
        beta=0.04,
        repetition_penalty=1.0,  # CRITICAL: Must be 1.0 to match base policy logprobs
        # vLLM configuration
        use_vllm=True,
        vllm_gpu_memory_utilization=0.5,
        vllm_model_impl="transformers",
        vllm_mode=vllm_mode,
        optim="paged_adamw_8bit",
        gradient_checkpointing=True,
        logging_steps=10,
        save_steps=200,
        save_strategy="steps",
        eval_strategy="no",
        log_completions=False,  # avoid Rich library bugs
        num_completions_to_print=0,
        model_init_kwargs={
            "torch_dtype": torch.bfloat16,
            "attn_implementation": "flash_attention_2",
        },
        max_grad_norm=0.5,
        # Below enables GSPO:
        importance_sampling_level="sequence",
        mask_truncated_completions=False,
        vllm_importance_sampling_correction=True,
        **kwargs,
    )

    # Set vLLM server URL if in server mode
    if vllm_mode == "server" and vllm_server_base_url:
        config.vllm_server_base_url = vllm_server_base_url

    return config


def create_lora_config(
    lora_r: int = 32,
    lora_alpha: int = 64,
    lora_dropout: float = 0.05,
    target_modules: Optional[List[str]] = None,
) -> LoraConfig:
    """Create a LoRA configuration for parameter-efficient fine-tuning.

    Args:
        lora_r: LoRA rank (intrinsic dimension)
        lora_alpha: LoRA scaling factor
        lora_dropout: Dropout probability for LoRA layers
        target_modules: List of module names to apply LoRA to

    Returns:
        LoraConfig instance
    """
    if target_modules is None:
        # Default target modules for Gemma (all attention and MLP projections)
        target_modules = [
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ]

    lora_config = LoraConfig(
        r=lora_r,
        lora_alpha=lora_alpha,
        target_modules=target_modules,
        lora_dropout=lora_dropout,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
        modules_to_save=["embed_tokens", "lm_head"],
    )
    return lora_config


def main():
    parser = argparse.ArgumentParser(
        description="Train Gemma-3-4B with GRPO + vLLM colocate mode + Infinite-RL curriculum. "
        "NOTE: Requires vocabulary pre-expansion using train/model_expand.py"
    )

    # Model arguments
    parser.add_argument(
        "--model_name",
        type=str,
        default="jed351/Gemma3-4B-ChatVector_SFT-from-IT_and_IT",
        help="Model name or path (should be pre-expanded with special tokens)",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./grpo-gemma3-4b",
        help="Output directory for checkpoints",
    )

    # Training arguments
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=1e-5,
        help="Learning rate",
    )
    parser.add_argument(
        "--num_train_epochs",
        type=int,
        default=3,
        help="Number of training epochs",
    )
    parser.add_argument(
        "--per_device_train_batch_size",
        type=int,
        default=4,
        help="Batch size per device",
    )
    parser.add_argument(
        "--num_generations",
        type=int,
        default=4,
        help="Number of completions per prompt",
    )
    parser.add_argument(
        "--max_completion_length",
        type=int,
        default=768,
        help="Maximum completion length",
    )
    parser.add_argument(
        "--max_prompt_length",
        type=int,
        default=2048,
        help="Maximum prompt length (for vLLM context sizing - limits vLLM KV cache allocation)",
    )
    parser.add_argument(
        "--num_train_samples",
        type=int,
        default=None,
        help="Number of training samples to generate per refresh. "
        "If None, defaults to batch_size * gradient_accumulation_steps * 4 (enough for a few batches)",
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of steps to accumulate gradients before updating weights",
    )

    # vLLM mode arguments
    parser.add_argument(
        "--vllm_mode",
        type=str,
        choices=["server", "colocate"],
        default="colocate",
        help="vLLM mode: 'server' for external vLLM server, 'colocate' for in-process vLLM",
    )
    parser.add_argument(
        "--vllm_server_base_url",
        type=str,
        default=None,
        help="Base URL for vLLM server mode (required when --vllm-mode=server)",
    )

    # Curriculum arguments
    parser.add_argument(
        "--warmup_step",
        type=int,
        default=32,
        help="Number of warmup steps (level 0 only)",
    )
    parser.add_argument(
        "--window_size",
        type=int,
        default=50,
        help="Sliding window size for success rate tracking",
    )
    parser.add_argument(
        "--success_rate_threshold",
        type=float,
        default=0.7,
        help="Success rate threshold for advancing difficulty",
    )
    parser.add_argument(
        "--variance_threshold",
        type=float,
        default=0.15,
        help="Variance threshold for stability check",
    )
    parser.add_argument(
        "--demote_threshold",
        type=float,
        default=0.4,
        help="Success rate threshold for demoting difficulty (demote if success_rate < demote_threshold)",
    )
    parser.add_argument(
        "--reasoning_language",
        type=str,
        default="en",
        choices=["en", "yue", "zh"],
        help="Language for reasoning (CoT) in <think> tags (default: en). Options: en=English, yue=Cantonese, zh=Mandarin",
    )
    parser.add_argument(
        "--reasoning_template",
        action="store_true",
        help="Model uses a reasoning chat template that injects <think> automatically. "
        "When enabled, skip checking for the opening <think> tag (closing </think> is still required).",
    )
    parser.add_argument(
        "--log_curriculum_steps",
        type=int,
        default=10,
        help="Log curriculum stats to W&B every N training steps (default: 10)",
    )

    # LLM Judge arguments
    parser.add_argument(
        "--use_llm_judge",
        action="store_true",
        help="Enable LLM Judge for auxiliary reward scoring (requires running sglang server)",
    )
    parser.add_argument(
        "--llm_judge_host",
        type=str,
        default="localhost",
        help="Host address of the sglang LLM Judge server (default: localhost)",
    )
    parser.add_argument(
        "--llm_judge_port",
        type=int,
        default=8000,
        help="Port of the sglang LLM Judge server (default: 8000)",
    )
    parser.add_argument(
        "--llm_judge_model",
        type=str,
        default="Skywork/Skywork-Reward-V2-Qwen3-4B",
        help="Model name for LLM Judge (default: Skywork/Skywork-Reward-V2-Qwen3-4B)",
    )

    # LoRA/PEFT arguments
    parser.add_argument(
        "--use_lora",
        action="store_true",
        help="Use LoRA (Low-Rank Adaptation) for parameter-efficient fine-tuning",
    )
    parser.add_argument(
        "--lora_r",
        type=int,
        default=32,
        help="LoRA rank (intrinsic dimension)",
    )
    parser.add_argument(
        "--lora_alpha",
        type=int,
        default=64,
        help="LoRA scaling factor",
    )
    parser.add_argument(
        "--lora_dropout",
        type=float,
        default=0.05,
        help="Dropout probability for LoRA layers",
    )

    # Quantization arguments
    parser.add_argument(
        "--load_in_4bit",
        action="store_true",
        help="Load model in 4-bit quantization using bitsandbytes",
    )
    parser.add_argument(
        "--load_in_8bit",
        action="store_true",
        help="Load model in 8-bit quantization using bitsandbytes",
    )
    parser.add_argument(
        "--bnb_4bit_compute_dtype",
        type=str,
        default="bf16",
        choices=["fp16", "bf16", "fp32"],
        help="Compute dtype for 4-bit quantization (default: bf16)",
    )
    parser.add_argument(
        "--bnb_4bit_quant_type",
        type=str,
        default="nf4",
        choices=["fp4", "nf4"],
        help="Quantization type for 4-bit (default: nf4)",
    )
    parser.add_argument(
        "--bnb_4bit_use_double_quant",
        action="store_true",
        help="Use double quantization for 4-bit (reduces memory further)",
    )

    # GRPO loss type arguments
    parser.add_argument(
        "--dapo",
        action="store_true",
        help="Use DAPO (Decoupled Advantage Policy Optimization) loss",
    )

    # Training control arguments
    parser.add_argument(
        "--max_steps",
        type=int,
        default=None,
        help="Maximum number of training steps (if set, overrides num_train_epochs). "
        "Dataset size will be calculated to match this.",
    )

    # W&B arguments
    parser.add_argument(
        "--use_wandb",
        action="store_true",
        help="Enable Weights & Biases (W&B) logging",
    )
    parser.add_argument(
        "--wandb_project",
        type=str,
        default="infinite-rl-grpo",
        help="W&B project name",
    )
    parser.add_argument(
        "--wandb_entity",
        type=str,
        default=None,
        help="W&B entity (team/user) name",
    )
    parser.add_argument(
        "--wandb_run_name",
        type=str,
        default=None,
        help="Custom run name for W&B",
    )

    args = parser.parse_args()

    # Validate quantization arguments
    if args.load_in_4bit and args.load_in_8bit:
        parser.error(
            "Cannot use both --load_in_4bit and --load_in_8bit. Choose one quantization method."
        )

    # Validate vLLM mode arguments
    if args.vllm_mode == "server" and args.vllm_server_base_url is None:
        parser.error("--vllm_server_base_url is required when --vllm-mode=server")

    # Calculate dataset size based on training configuration
    # This ensures we generate exactly the number of samples needed
    effective_batch_size = (
        args.per_device_train_batch_size * args.gradient_accumulation_steps
    )

    if args.max_steps is not None:
        # Use max_steps to calculate total samples needed
        num_train_samples = args.max_steps * effective_batch_size
        print(f"\nCalculated dataset size: {num_train_samples} samples")
        print(
            f"  (max_steps={args.max_steps} * effective_batch={effective_batch_size})"
        )
    else:
        # Use num_train_epochs to calculate samples per epoch
        # Default to a reasonable number of steps per epoch (e.g., 100)
        steps_per_epoch = 100
        num_train_samples = steps_per_epoch * effective_batch_size
        print(f"\nCalculated dataset size: {num_train_samples} samples per epoch")
        print(
            f"  (steps_per_epoch={steps_per_epoch} * effective_batch={effective_batch_size})"
        )
        print(f"  Training for {args.num_train_epochs} epochs")

    # Initialize W&B if enabled
    if args.use_wandb:
        wandb_config = {
            "model": args.model_name,
            "learning_rate": args.learning_rate,
            "num_train_epochs": args.num_train_epochs,
            "batch_size": args.per_device_train_batch_size,
            "num_generations": args.num_generations,
            "max_completion_length": args.max_completion_length,
            "use_dapo": args.dapo,
            "use_lora": args.use_lora,
            "use_vllm": True,  # Always enabled for train2.py
            "load_in_4bit": args.load_in_4bit,
            "load_in_8bit": args.load_in_8bit,
            "window_size": args.window_size,
            "success_rate_threshold": args.success_rate_threshold,
        }

        wandb.init(
            project=args.wandb_project,
            entity=args.wandb_entity,
            name=args.wandb_run_name,
            config=wandb_config,
            tags=["grpo", "curriculum", "infinite-rl", "vllm-colocate"],
        )
        print("\n✓ W&B initialized for run tracking")

    # Setup paths
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("Infinite-RL + GRPO Training (vLLM Colocate Mode)")
    print("=" * 80)

    # Initialize curriculum
    print("\n1. Initializing Infinite-RL curriculum...")
    curriculum_config = InfiniteRLConfig(
        warmup_step=args.warmup_step,
        window_size=args.window_size,
        success_rate_threshold=args.success_rate_threshold,
        variance_threshold=args.variance_threshold,
        demote_threshold=args.demote_threshold,
        num_generations=args.num_generations,
        reasoning_language=args.reasoning_language,
        reasoning_template=args.reasoning_template,
        use_llm_judge=args.use_llm_judge,
        llm_judge_host=args.llm_judge_host,
        llm_judge_port=args.llm_judge_port,
        llm_judge_model=args.llm_judge_model,
        log_file=str(output_dir / "curriculum_learning_log.jsonl"),
    )
    curriculum = create_curriculum(curriculum_config)
    print(f"   ✓ Curriculum initialized")
    print(f"   - Learning stats: {curriculum.get_learning_stats()}")

    # Load model and tokenizer
    print(f"\n2. Loading model & tokenizer: {args.model_name}...")

    # Prepare quantization config if requested
    quantization_config = None
    if args.load_in_4bit or args.load_in_8bit:
        from transformers import BitsAndBytesConfig

        compute_dtype = torch.bfloat16
        if args.bnb_4bit_compute_dtype == "fp16":
            compute_dtype = torch.float16
        elif args.bnb_4bit_compute_dtype == "fp32":
            compute_dtype = torch.float32

        quantization_config = BitsAndBytesConfig(
            load_in_4bit=args.load_in_4bit,
            load_in_8bit=args.load_in_8bit,
            bnb_4bit_compute_dtype=compute_dtype,
            bnb_4bit_quant_type=args.bnb_4bit_quant_type,
            bnb_4bit_use_double_quant=args.bnb_4bit_use_double_quant,
        )
        print(f"   - Using quantization: {'4-bit' if args.load_in_4bit else '8-bit'}")
        print(f"   - Compute dtype: {args.bnb_4bit_compute_dtype}")
        if args.load_in_4bit:
            print(f"   - 4-bit quant type: {args.bnb_4bit_quant_type}")
            print(f"   - Double quant: {args.bnb_4bit_use_double_quant}")

    # In colocate mode, set device_map to None to avoid memory conflicts
    # For quantization, use "auto" device_map for proper GPU placement
    device_map = None if not (args.load_in_4bit or args.load_in_8bit) else "auto"

    model = Gemma3ForCausalLM.from_pretrained(
        args.model_name,
        torch_dtype=torch.bfloat16,
        device_map=device_map,
        quantization_config=quantization_config,
        attn_implementation="flash_attention_2",
    )
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print(f"   ✓ Model loaded: {model.config.model_type}")
    print(f"   - Parameters: {sum(p.numel() for p in model.parameters()):,}")
    if args.load_in_4bit or args.load_in_8bit:
        print(f"   - Quantization: {'4-bit' if args.load_in_4bit else '8-bit'}")
    # print(f"   ✓ Tokenizer loaded with vocab size: {len(tokenizer)}")
    print(f"   - Model config vocab_size: {model.config.vocab_size}")

    # # Verify vocab consistency
    if len(tokenizer) != model.config.vocab_size:
        print(f"\n   ⚠️  WARNING: Vocab size mismatch!")
        print(
            f"     Tokenizer: {len(tokenizer)}, Model config: {model.config.vocab_size}"
        )
        print(f"     Run train/model_expand.py to fix this issue.")

    # Apply LoRA if requested
    if args.use_lora:
        print(f"\n   Applying LoRA (Parameter-Efficient Fine-Tuning)...")
        lora_config = create_lora_config(
            lora_r=args.lora_r,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
        )
        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()
        print(f"   ✓ LoRA applied successfully")

    # Build dynamic dataset that generates prompts on-demand
    print(f"\n3. Creating dynamic curriculum dataset...")
    train_dataset = build_dynamic_dataset(
        curriculum,
        num_samples=num_train_samples,
    )
    print(f"   ✓ Dynamic dataset created (size: {len(train_dataset)} samples)")
    print(f"   - Prompts generated on-demand from current curriculum level")
    print(f"   - Dataset size matches training batch configuration")
    print(f"   - No pre-generation or refresh needed!")

    # Create reward function
    print("\n4. Creating Infinite-RL-based reward function...")
    reward_func = create_curriculum_reward_func(curriculum)
    print(f"   ✓ Reward function created")

    # Setup training configuration
    print("\n5. Setting up GRPO training configuration (vLLM colocate mode)...")
    grpo_kwargs = {}
    if args.dapo:
        grpo_kwargs["loss_type"] = "dapo"
    if args.max_steps is not None:
        grpo_kwargs["max_steps"] = args.max_steps

    training_args = setup_training_args(
        output_dir=args.output_dir,
        learning_rate=args.learning_rate,
        num_train_epochs=args.num_train_epochs,
        per_device_train_batch_size=args.per_device_train_batch_size,
        num_generations=args.num_generations,
        max_completion_length=args.max_completion_length,
        max_prompt_length=args.max_prompt_length,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        warmup_steps=args.warmup_step,
        vllm_mode=args.vllm_mode,
        vllm_server_base_url=args.vllm_server_base_url,
        **grpo_kwargs,
    )
    print(f"   ✓ Training configuration ready")
    print(f"   - Batch size (per device): {training_args.per_device_train_batch_size}")
    print(f"   - Number of generations: {training_args.num_generations}")
    print(f"   - Max completion length: {training_args.max_completion_length}")
    print(f"   - Max prompt length: {args.max_prompt_length}")
    print(
        f"   - vLLM max_model_len: {args.max_prompt_length + args.max_completion_length} tokens (overrides model default)"
    )
    print(f"   - vLLM GPU memory utilization: 50%")
    print(f"   - Learning rate: {training_args.learning_rate}")
    print(f"   - Warmup steps: {training_args.warmup_steps}")
    print(f"   - vLLM mode: {args.vllm_mode}")
    if args.vllm_mode == "server":
        print(f"   - vLLM server URL: {args.vllm_server_base_url}")
    if args.max_steps:
        print(f"   - Max steps: {args.max_steps}")
    else:
        print(f"   - Num epochs: {args.num_train_epochs}")
    if args.dapo:
        print(f"   - Loss type: DAPO (Decoupled Advantage Policy Optimization)")

    # Initialize trainer
    print("\n6. Initializing GRPO trainer...")

    # Setup callbacks for curriculum logging
    callbacks = []
    curriculum_callback = CurriculumLoggingCallback(
        curriculum=curriculum,
        log_every_n_steps=args.log_curriculum_steps,
    )
    callbacks.append(curriculum_callback)
    print(
        f"   - Added curriculum logging callback (log every {args.log_curriculum_steps} steps)"
    )

    trainer = GRPOTrainer(
        model=model,
        reward_funcs=[reward_func],  # Pass as list
        args=training_args,
        train_dataset=train_dataset,
        processing_class=tokenizer,
        callbacks=callbacks if callbacks else None,
    )
    print(f"   ✓ Trainer initialized")

    # Save configuration
    config_dict = {
        "model_name": args.model_name,
        "vllm_mode": args.vllm_mode,
        "vllm_server_base_url": (
            args.vllm_server_base_url if args.vllm_mode == "server" else None
        ),
        "load_in_4bit": args.load_in_4bit,
        "load_in_8bit": args.load_in_8bit,
        "quantization_config": (
            {
                "bnb_4bit_compute_dtype": args.bnb_4bit_compute_dtype,
                "bnb_4bit_quant_type": args.bnb_4bit_quant_type,
                "bnb_4bit_use_double_quant": args.bnb_4bit_use_double_quant,
            }
            if args.load_in_4bit
            else None
        ),
        "curriculum_config": asdict(curriculum_config),
        "training_config": training_args.to_dict(),
        "dataset_size": len(train_dataset),
    }

    # Convert to JSON-serializable format (handles dtype objects)
    config_dict = make_json_serializable(config_dict)

    # Create the output directory if it doesn't exist
    output_dir.mkdir(parents=True, exist_ok=True)

    with open(output_dir / "training_config.json", "w") as f:
        json.dump(config_dict, f, indent=2)
    print(f"   ✓ Configuration saved to {output_dir / 'training_config.json'}")

    # Start training
    print("\n7. Starting GRPO training...")
    print("=" * 80)

    trainer.train()
    print("=" * 80)
    trainer.save_model(str(output_dir / "final_model"))
    print(f"✓ Final model saved to {output_dir / 'final_model'}")

    # Print final curriculum stats
    final_stats = curriculum.get_learning_stats()
    judge_stats = curriculum.get_judge_scores()
    print(f"\nFinal Curriculum Statistics:")
    print(f"  - Current level: {final_stats['current_level']}")
    print(f"  - Sliding window stats: {final_stats['sliding_window_stats']}")
    print(
        f"  - Judge scores: avg={judge_stats['avg_judge_score']:.3f}, count={judge_stats['judge_score_count']}"
    )

    # Log final stats to W&B
    if args.use_wandb and wandb.run is not None:
        final_wandb_logs = {
            "curriculum/final_level": final_stats.get("current_level", 0),
        }

        # Log judge scores
        final_wandb_logs["judge/final_avg_score"] = judge_stats.get(
            "avg_judge_score", 0.0
        )
        final_wandb_logs["judge/final_min_score"] = judge_stats.get(
            "min_judge_score", 0.0
        )
        final_wandb_logs["judge/final_max_score"] = judge_stats.get(
            "max_judge_score", 0.0
        )
        final_wandb_logs["judge/final_score_count"] = judge_stats.get(
            "judge_score_count", 0
        )
        final_wandb_logs["judge/final_total_tasks_with_judge"] = judge_stats.get(
            "total_tasks_with_judge", 0
        )

        # Log final success rates and statistics
        sliding_window = final_stats.get("sliding_window_stats", {})

        # Log aggregated stats (these are always at top level)
        if "mean_success_rate" in sliding_window:
            final_wandb_logs["curriculum/final_mean_success_rate"] = sliding_window.get(
                "mean_success_rate", 0
            )
            final_wandb_logs["curriculum/final_mean_variance"] = sliding_window.get(
                "mean_variance", 0
            )
            final_wandb_logs["curriculum/final_total_samples"] = sliding_window.get(
                "samples", 0
            )

        # Log per-level stats
        for level_stat in sliding_window.get("by_level", []):
            level = level_stat.get("level", "unknown")
            success_rate = level_stat.get("success_rate", 0)
            variance = level_stat.get("variance", 0)
            samples = level_stat.get("samples", 0)

            final_wandb_logs[f"final_stats/level_{level}/success_rate"] = success_rate
            final_wandb_logs[f"final_stats/level_{level}/variance"] = variance
            final_wandb_logs[f"final_stats/level_{level}/samples"] = samples

        wandb.log(final_wandb_logs)
        wandb.finish()
        print(f"\n✓ Final stats logged to W&B")

    # Save final stats
    final_stats_with_judge = {**final_stats, "judge_scores": judge_stats}
    with open(output_dir / "final_stats.json", "w") as f:
        json.dump(final_stats_with_judge, f, indent=2, default=str)
    print(f"✓ Final stats saved to {output_dir / 'final_stats.json'}")


if __name__ == "__main__":
    main()
