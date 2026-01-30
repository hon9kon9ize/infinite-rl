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

import json
import argparse
from pathlib import Path
from typing import Optional, List, Dict, Any, Callable
from dataclasses import dataclass, asdict

import torch
import wandb
from transformers import (
    Gemma3ForCausalLM,
    AutoTokenizer,
    TrainerCallback,
)
from datasets import Dataset
from trl import GRPOTrainer, GRPOConfig
from peft import LoraConfig, get_peft_model, TaskType

# Import infinite-rl components
from infinite_rl.curriculum import CurriculumLearning
from infinite_rl.dynamic_dataset import DynamicCurriculumDataset

import functools
from vllm import LLM


# --- THE FIX ---
original_init = LLM.__init__


@functools.wraps(original_init)
def patched_init(self, *args, **kwargs):
    # Disable the code path that tries to 'profile' multimodal inputs
    kwargs["skip_mm_profiling"] = True
    # Optional: ensure it stays in eager mode for better memory control
    kwargs["enforce_eager"] = True
    return original_init(self, *args, **kwargs)


LLM.__init__ = patched_init
# ----------------


@dataclass
class InfiniteRLConfig:
    """Configuration for Infinite-RL curriculum integration."""

    # Curriculum parameters
    window_size: int = 50
    success_rate_threshold: float = 0.8
    variance_threshold: float = 0.05
    demote_threshold: float = 0.4
    warmup_step: int = 32
    level_change_cooldown: int = 5
    reflective_learning_rate: float = 0.2
    num_generations: int = 4

    # Reward function parameters
    timeout: int = 10
    answer_tag: str = "answer"
    think_tag: str = "think"

    # Auxiliary rewards
    use_format: bool = True
    use_reasoning_steps: bool = True
    use_length: bool = True
    use_lang_consistency: bool = True
    use_repetition: bool = True
    aux_weight: float = 0.3

    # Output
    log_file: Optional[str] = "curriculum_learning_log.jsonl"


def create_curriculum(config: InfiniteRLConfig) -> CurriculumLearning:
    """Initialize the curriculum learning system from Infinite-RL.

    Args:
        config: Infinite-RL configuration

    Returns:
        Initialized CurriculumLearning instance
    """
    curriculum = CurriculumLearning(
        timeout=config.timeout,
        answer_tag=config.answer_tag,
        think_tag=config.think_tag,
        aux_weight=config.aux_weight,
        use_format=config.use_format,
        use_reasoning_steps=config.use_reasoning_steps,
        use_length=config.use_length,
        use_lang_consistency=config.use_lang_consistency,
        use_repetition=config.use_repetition,
        window_size=config.window_size,
        success_rate_threshold=config.success_rate_threshold,
        variance_threshold=config.variance_threshold,
        demote_threshold=config.demote_threshold,
        warmup_step=config.warmup_step,
        reflective_learning_rate=config.reflective_learning_rate,
        level_change_cooldown=config.level_change_cooldown,
        num_generations=config.num_generations,
        log_file=config.log_file,
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

    The reward function wraps Infinite-RL's curriculum reward computation
    and returns scores in the range [0, 1] as expected by GRPO.

    Args:
        curriculum: Initialized CurriculumLearning instance

    Returns:
        A callable that computes rewards for completions
    """

    def reward_func(
        prompts: List[str],
        completions: List[str],
        task_metadata: Optional[List[Dict[str, Any]]] = None,
        **kwargs,
    ) -> List[float]:
        """Compute rewards for completions using Infinite-RL's curriculum.

        Args:
            prompts: List of prompts (can be strings or list-of-dicts from TRL)
            completions: List of model completions
            task_metadata: Metadata for each task (task_id, task_type, etc.)
            **kwargs: Additional arguments from trainer (ignored)

        Returns:
            List of reward scores in range [0, 1]
        """
        rewards = []

        for idx, completion in enumerate(completions):
            try:
                # TRL's task_metadata is often a list of dicts corresponding to the prompts.
                # If num_generations > 1, the metadata is repeated for each generation in the group.
                meta = (
                    task_metadata[idx]
                    if task_metadata and idx < len(task_metadata)
                    else {}
                )
                task_id = meta.get("task_id", f"task_{idx}")

                # Extract content from TRL's chat format
                # Completions come as: [{'role': 'assistant', 'content': '...'}]
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
                        f"Warning: Completion for task {idx} is not a string after extraction. Type: {type(completion_text)}"
                    )
                    rewards.append(0.0)
                    continue

                # Compute reward using curriculum
                reward = curriculum.compute_reward(task_id, completion_text)
                rewards.append(float(reward))

            except Exception as e:
                print(f"Warning: Error computing reward for task {idx}: {e}")
                rewards.append(0.0)

        return rewards

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

                # Log success rates by task type
                sliding_window = stats.get("sliding_window_stats", {})

                if "by_task_type" in sliding_window:
                    wandb_logs["curriculum/mean_success_rate"] = sliding_window.get(
                        "mean_success_rate", 0
                    )
                    wandb_logs["curriculum/mean_variance"] = sliding_window.get(
                        "mean_variance", 0
                    )
                    wandb_logs["curriculum/total_samples"] = sliding_window.get(
                        "samples", 0
                    )

                    for task_stat in sliding_window.get("by_task_type", []):
                        task_type = task_stat.get("task_type", "unknown")
                        success_rate = task_stat.get("success_rate", 0)
                        variance = task_stat.get("variance", 0)
                        samples = task_stat.get("samples", 0)

                        wandb_logs[f"curriculum/{task_type}/success_rate"] = (
                            success_rate
                        )
                        wandb_logs[f"curriculum/{task_type}/variance"] = variance
                        wandb_logs[f"curriculum/{task_type}/samples"] = samples
                else:
                    for task_type, window_stats in sliding_window.items():
                        if isinstance(window_stats, dict):
                            success_rate = window_stats.get("success_rate", 0)
                            variance = window_stats.get("variance", 0)
                            samples = window_stats.get("samples", 0)

                            wandb_logs[f"curriculum/{task_type}/success_rate"] = (
                                success_rate
                            )
                            wandb_logs[f"curriculum/{task_type}/variance"] = variance
                            wandb_logs[f"curriculum/{task_type}/samples"] = samples

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
    **kwargs,
) -> GRPOConfig:
    """Setup GRPO training configuration for vLLM colocate mode.

    Args:
        output_dir: Output directory for checkpoints
        learning_rate: Learning rate for training
        num_train_epochs: Number of training epochs
        per_device_train_batch_size: Batch size per device
        num_generations: Number of completions per prompt
        max_completion_length: Maximum completion length
        max_prompt_length: Maximum prompt length (for vLLM context size)
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
        gradient_accumulation_steps=gradient_accumulation_steps,
        temperature=1.0,
        top_p=1.0,
        # vLLM COLOCATE mode is always enabled
        use_vllm=True,
        vllm_gpu_memory_utilization=0.5,
        vllm_model_impl="transformers",
        vllm_mode="colocate",
        optim="paged_adamw_8bit",
        gradient_checkpointing=True,
        logging_steps=10,
        save_steps=100,
        save_strategy="steps",
        eval_strategy="no",
        log_completions=True,
        num_completions_to_print=5,
        **kwargs,
    )
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
        default=0.8,
        help="Success rate threshold for advancing difficulty",
    )
    parser.add_argument(
        "--variance_threshold",
        type=float,
        default=0.05,
        help="Variance threshold for stability check",
    )
    parser.add_argument(
        "--log_curriculum_steps",
        type=int,
        default=10,
        help="Log curriculum stats to W&B every N training steps (default: 10)",
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
        num_generations=args.num_generations,
        log_file=str(output_dir / "curriculum_learning_log.jsonl"),
    )
    curriculum = create_curriculum(curriculum_config)
    print(f"   ✓ Curriculum initialized")
    print(f"   - Learning stats: {curriculum.get_learning_stats()}")

    # Load model and tokenizer
    print(f"\n2. Loading model & tokenizer: {args.model_name}...")

    # In colocate mode, set device_map to None to avoid memory conflicts
    model = Gemma3ForCausalLM.from_pretrained(
        args.model_name,
        torch_dtype=torch.bfloat16,
        device_map=None,
    )
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print(f"   ✓ Model loaded: {model.config.model_type}")
    print(f"   - Parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"   ✓ Tokenizer loaded with vocab size: {len(tokenizer)}")
    print(f"   - Model config vocab_size: {model.config.vocab_size}")

    # Verify vocab consistency
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
    print(f"   - vLLM mode: colocate (in-process)")
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
        reward_funcs=reward_func,
        args=training_args,
        train_dataset=train_dataset,
        processing_class=tokenizer,
        callbacks=callbacks if callbacks else None,
    )
    print(f"   ✓ Trainer initialized")

    # Save configuration
    config_dict = {
        "model_name": args.model_name,
        "vllm_mode": "colocate",
        "curriculum_config": asdict(curriculum_config),
        "training_config": training_args.to_dict(),
        "dataset_size": len(train_dataset),
    }

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
    print(f"\nFinal Curriculum Statistics:")
    print(f"  - Current level: {final_stats['current_level']}")
    print(f"  - Sliding window stats: {final_stats['sliding_window_stats']}")

    # Log final stats to W&B
    if args.use_wandb and wandb.run is not None:
        final_wandb_logs = {
            "curriculum/final_level": final_stats.get("current_level", 0),
        }

        # Log final success rates and statistics
        sliding_window = final_stats.get("sliding_window_stats", {})

        # Handle both aggregated and per-task-type stats
        if "by_task_type" in sliding_window:
            # Aggregated stats with per-task-type breakdown
            final_wandb_logs["curriculum/final_mean_success_rate"] = sliding_window.get(
                "mean_success_rate", 0
            )
            final_wandb_logs["curriculum/final_mean_variance"] = sliding_window.get(
                "mean_variance", 0
            )
            final_wandb_logs["curriculum/final_total_samples"] = sliding_window.get(
                "samples", 0
            )

            for task_stat in sliding_window.get("by_task_type", []):
                task_type = task_stat.get("task_type", "unknown")
                success_rate = task_stat.get("success_rate", 0)
                variance = task_stat.get("variance", 0)
                samples = task_stat.get("samples", 0)

                final_wandb_logs[f"final_stats/{task_type}/success_rate"] = success_rate
                final_wandb_logs[f"final_stats/{task_type}/variance"] = variance
                final_wandb_logs[f"final_stats/{task_type}/samples"] = samples
        else:
            # Single task type stats
            for task_type, window_stats in sliding_window.items():
                if isinstance(window_stats, dict) and task_type not in [
                    "mean_success_rate",
                    "mean_variance",
                ]:
                    success_rate = window_stats.get("success_rate", 0)
                    variance = window_stats.get("variance", 0)
                    samples = window_stats.get("samples", 0)

                    final_wandb_logs[f"final_stats/{task_type}/success_rate"] = (
                        success_rate
                    )
                    final_wandb_logs[f"final_stats/{task_type}/variance"] = variance
                    final_wandb_logs[f"final_stats/{task_type}/samples"] = samples

        wandb.log(final_wandb_logs)
        wandb.finish()
        print(f"\n✓ Final stats logged to W&B")

    # Save final stats
    with open(output_dir / "final_stats.json", "w") as f:
        json.dump(final_stats, f, indent=2, default=str)
    print(f"✓ Final stats saved to {output_dir / 'final_stats.json'}")


if __name__ == "__main__":
    main()
