import os
import pandas as pd
import google.generativeai as genai
import json
import random
from dotenv import load_dotenv
from .prompts import SYSTEM_PROMPT, TYPE_PROMPTS
from .parser import ExampleParser
from .reward_functions.coding import CodingRewardFunction
from .reward_functions.math import MathRewardFunction
from .reward_functions.summarization import SummarizationRewardFunction
from .reward_functions.html import HtmlRewardFunction

load_dotenv()


def get_reward_function(task_type, timeout=5):
    """Retrieve the appropriate reward function for a task type."""
    if task_type in ["python", "javascript", "rust", "cpp", "java", "coding"]:
        fn = CodingRewardFunction(task_name=task_type, timeout=timeout)
        if task_type in ["javascript", "js"]:
            fn.set_language("javascript")
        elif task_type in ["rust"]:
            fn.set_language("rust")
        elif task_type in ["cpp", "c++"]:
            fn.set_language("cpp")
        elif task_type in ["java"]:
            fn.set_language("java")
        return fn
    elif task_type == "math":
        return MathRewardFunction(task_name="math", timeout=timeout)
    elif task_type == "summarization":
        return SummarizationRewardFunction(task_name="summarization", timeout=timeout)
    elif task_type == "html":
        return HtmlRewardFunction(task_name="html", timeout=timeout)
    return None


def generate_dataset(
    model_name, num_samples, out_dir, save_every=10, max_retries=5, timeout=5
):
    # Enforce specified distribution: 80% coding, 10% html, 10% summarization
    distribution = {"coding": 0.8, "html": 0.1, "summarization": 0.1}

    # Calculate counts based on distribution
    type_counts = {t: int(num_samples * weight) for t, weight in distribution.items()}

    # Ensure we reach total samples by adding remainder to the main type (coding)
    total_calculated = sum(type_counts.values())
    if total_calculated < num_samples:
        type_counts["coding"] += num_samples - total_calculated

    print(f"Generating {num_samples} samples using {model_name}...")
    print(f"Distribution: {type_counts}")

    dataset = []
    failed_dataset = []
    all_successful_prompts = []

    # Load initial seed prompts from examples directory for diversity
    examples_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "examples")
    if os.path.exists(examples_dir):
        initial_examples = ExampleParser.get_all_examples(examples_dir)
        for ex in initial_examples.values():
            if ex.get("prompt"):
                all_successful_prompts.append(ex["prompt"])

    # Configure Gemini
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise ValueError("GOOGLE_API_KEY not found in environment variables")

    genai.configure(api_key=api_key)
    model = genai.GenerativeModel(
        model_name=model_name, system_instruction=SYSTEM_PROMPT
    )

    for t, count in type_counts.items():
        if count <= 0:
            continue

        print(f"Generating {count} samples for type: {t}")

        type_prompt_base = TYPE_PROMPTS.get(t, f"Generate a sample for {t}")
        reward_fn = get_reward_function(t, timeout=timeout)
        if reward_fn:
            reward_fn.initialize()

        current_count = 0
        consecutive_retries = 0
        fatal_error = False

        while current_count < count:
            if consecutive_retries >= max_retries:
                print(
                    f"  [FATAL] Maximum retries ({max_retries}) reached for type {t}. Stopping process."
                )
                fatal_error = True
                break

            try:
                # Diversity: Pick a random seed prompt from history if available
                seed_context = ""
                if all_successful_prompts:
                    random_seed = random.choice(all_successful_prompts)
                    seed_context = f'\n\nHere is a prompt we previously generated: "{random_seed}". Please generate something DIFFERENT and more complex than this.'

                # Combine base prompt with seed context
                full_query = (
                    f"Generate a new {t} example. {type_prompt_base}{seed_context}"
                )

                response = model.generate_content(full_query)

                raw_text = response.text
                parsed = ExampleParser.parse_text(raw_text)

                # Validation
                quality_score = None
                if reward_fn:
                    kwargs = {}
                    if t == "summarization":
                        kwargs["original_document"] = parsed["prompt"]

                    score = reward_fn.compute_reward(
                        parsed["response"], parsed["answer"], **kwargs
                    )
                    quality_score = (score.format_score + score.correctness_score) / 2.0

                    if quality_score < 0.8:
                        print(f"  [Retry] Sample quality too low ({quality_score:.2f})")
                        print(f"    - Format Score: {score.format_score:.2f}")
                        print(f"    - Correctness Score: {score.correctness_score:.2f}")
                        print(f"    - Error: {score.error_msg}")

                        # Log failed sample
                        failed_dataset.append(
                            {
                                "type": t,
                                "prompt": parsed["prompt"],
                                "answer": parsed["answer"],
                                "response": parsed["response"],
                                "format_score": score.format_score,
                                "correctness_score": score.correctness_score,
                                "error_msg": score.error_msg,
                                "raw_text": raw_text,
                            }
                        )

                        # Periodic save of failed dataset
                        if len(failed_dataset) % save_every == 0:
                            fdf = pd.DataFrame(failed_dataset)
                            fdf.to_csv(
                                os.path.join(out_dir, "failed_dataset.csv"), index=False
                            )

                        consecutive_retries += 1
                        continue

                sample = {
                    "type": t,
                    "prompt": parsed["prompt"],
                    "answer": parsed["answer"],
                    "response": parsed["response"],
                    "quality_score": quality_score,
                }
                dataset.append(sample)
                all_successful_prompts.append(parsed["prompt"])

                current_count += 1
                consecutive_retries = 0
                print(f"  [OK] Generated sample {current_count}/{count}")

                # Periodic save
                if len(dataset) % save_every == 0:
                    df = pd.DataFrame(dataset)
                    df.to_csv(os.path.join(out_dir, "dataset.csv"), index=False)
                    print(f"  [INFO] Saved progress ({len(dataset)} samples)")

            except Exception as e:
                print(f"Error generating sample for {t}: {e}")
                break

        if fatal_error:
            break

    # Save final results
    df = pd.DataFrame(dataset)
    df.to_csv(os.path.join(out_dir, "dataset.csv"), index=False)

    if failed_dataset:
        fdf = pd.DataFrame(failed_dataset)
        fdf.to_csv(os.path.join(out_dir, "failed_dataset.csv"), index=False)

    print(f"Dataset saved to {out_dir}")
