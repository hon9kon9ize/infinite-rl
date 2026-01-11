import os
import pandas as pd
import google.generativeai as genai
import json
import random
from tqdm import tqdm
from dotenv import load_dotenv
from .prompts import SYSTEM_PROMPT, TYPE_PROMPTS, RECTIFY_PROMPT
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
    model_name,
    num_samples,
    out_dir,
    save_every=10,
    max_retries=5,
    timeout=5,
    task_dist="0.5,0.1,0.3,0.1",
):
    # Create debug directory if it doesn't exist
    debug_dir = os.path.join(out_dir, "debug_prompts")
    if not os.path.exists(debug_dir):
        os.makedirs(debug_dir)

    # Parse task distribution: [code, html, math, summarization]
    try:
        dist_values = [float(v) for v in task_dist.split(",")]
        if len(dist_values) != 4:
            raise ValueError("task_dist must contain exactly 4 values")
    except Exception as e:
        print(f"Error parsing task_dist: {e}. Using default.")
        dist_values = [0.5, 0.1, 0.3, 0.1]

    distribution = {
        "coding": dist_values[0],
        "html": dist_values[1],
        "math": dist_values[2],
        "summarization": dist_values[3],
    }

    # Calculate counts based on distribution
    type_counts = {t: int(num_samples * weight) for t, weight in distribution.items()}

    # Ensure we reach total samples by adding remainder to the main type (coding)
    total_calculated = sum(type_counts.values())
    if total_calculated < num_samples:
        # Find the type with non-zero weight to add remainder, default to coding
        active_types = [t for t, w in distribution.items() if w > 0]
        addition_type = active_types[0] if active_types else "coding"
        type_counts[addition_type] += num_samples - total_calculated

    print(f"Generating {num_samples} samples using {model_name}...")
    print(f"Distribution: {type_counts}")

    dataset = []
    failed_dataset = []
    # Distribution includes sub-types for coding
    type_successful_prompts = {
        "coding": [],
        "python": [],
        "javascript": [],
        "typescript": [],
        "cpp": [],
        "rust": [],
        "java": [],
        "math": [],
        "summarization": [],
        "html": [],
    }

    # Load initial seed prompts from examples directory for diversity
    examples_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "examples")
    if os.path.exists(examples_dir):
        # We'll map filenames to task types
        file_to_type = {
            "PYTHON": "coding",
            "JAVASCRIPT": "javascript",
            "TYPESCRIPT": "typescript",
            "CPP": "cpp",
            "RUST": "rust",
            "JAVA": "java",
            "MATH": "math",
            "SUMMARIZATION": "summarization",
            "HTML": "html",
        }
        for filename in os.listdir(examples_dir):
            if filename.endswith(".md"):
                file_stem = filename.replace(".md", "").upper()
                task_t = file_to_type.get(file_stem)
                if task_t:
                    try:
                        with open(os.path.join(examples_dir, filename), "r") as f:
                            content = f.read()
                        parsed = ExampleParser.parse_text(content)
                        if parsed.get("prompt"):
                            type_successful_prompts[task_t].append(parsed["prompt"])
                            # also add to 'coding' if it's a specific language
                            if (
                                task_t
                                in [
                                    "python",
                                    "javascript",
                                    "typescript",
                                    "cpp",
                                    "rust",
                                    "java",
                                ]
                                and "coding" in type_successful_prompts
                            ):
                                type_successful_prompts["coding"].append(
                                    parsed["prompt"]
                                )
                    except Exception as e:
                        print(f"Warning: Could not load seed example {filename}: {e}")

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

        # Task specific seed examples
        success_list = type_successful_prompts.get(t, [])
        initial_seeds = list(success_list)  # copy initial seeds from examples/

        type_prompt_base = TYPE_PROMPTS.get(t, f"Generate a sample for {t}")
        reward_fn = get_reward_function(t, timeout=timeout)
        if reward_fn:
            reward_fn.initialize()

        current_count = 0
        consecutive_retries = 0
        fatal_error = False

        pbar = tqdm(total=count, desc=f"Type: {t}")

        while current_count < count:
            if consecutive_retries >= max_retries:
                print(
                    f"  [FATAL] Maximum retries ({max_retries}) reached for type {t}. Stopping process."
                )
                fatal_error = True
                break

            try:
                # Diversity: Pick a random seed prompt from history IF it is from the SAME task type
                seed_context = ""
                # Combined list of initial seeds and newly successful prompts for this task
                current_seeds = initial_seeds + success_list
                if current_seeds:
                    random_seed = random.choice(current_seeds)
                    seed_context = f'\n\nHere is a prompt we previously generated: "{random_seed}". Please generate something DIFFERENT and more complex than this.'

                # Combine base prompt with seed context
                full_query = (
                    f"Generate a new {t} example. {type_prompt_base}{seed_context}"
                )

                # Log the prompt for debugging
                debug_file = os.path.join(
                    debug_dir, f"{t}_{current_count}_{consecutive_retries}.txt"
                )
                with open(debug_file, "w") as f:
                    f.write(f"--- SYSTEM PROMPT ---\n{SYSTEM_PROMPT}\n\n")
                    f.write(f"--- USER PROMPT (Task: {t}) ---\n{full_query}\n")

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
                        if score.error_msg:
                            print(f"    - Error: {score.error_msg}")

                        # Attempt Rectification
                        rectify_count = 0
                        max_rectify = 2
                        current_raw = raw_text
                        current_score = score

                        while rectify_count < max_rectify and (quality_score < 0.8):
                            rectify_count += 1
                            print(f"  [Rectify] Attempt {rectify_count} for {t}...")

                            error_info = f"Format Score: {current_score.format_score:.2f}, Correctness Score: {current_score.correctness_score:.2f}"
                            if current_score.error_msg:
                                error_info += (
                                    f"\nError details: {current_score.error_msg}"
                                )

                            rectify_query = RECTIFY_PROMPT.format(
                                error_info=error_info, current_raw=current_raw
                            )

                            try:
                                response = model.generate_content(rectify_query)
                                current_raw = response.text
                                parsed = ExampleParser.parse_text(current_raw)

                                # Re-evaluate
                                kwargs = {}
                                if t == "summarization":
                                    kwargs["original_document"] = parsed["prompt"]

                                current_score = reward_fn.compute_reward(
                                    parsed["response"], parsed["answer"], **kwargs
                                )
                                quality_score = (
                                    current_score.format_score
                                    + current_score.correctness_score
                                ) / 2.0
                                raw_text = (
                                    current_raw  # Update for logging if it fails again
                                )

                                if quality_score >= 0.8:
                                    print(
                                        f"  [Success] Rectified sample quality: {quality_score:.2f}"
                                    )
                                    break
                                else:
                                    print(
                                        f"  [Retry] Rectification failed (quality: {quality_score:.2f})"
                                    )
                            except Exception as re:
                                print(f"    - Rectification error: {re}")
                                break

                        if quality_score < 0.8:
                            # Log failed sample after all rectification attempts
                            failed_dataset.append(
                                {
                                    "type": t,
                                    "prompt": parsed["prompt"],
                                    "answer": parsed["answer"],
                                    "response": parsed["response"],
                                    "format_score": current_score.format_score,
                                    "correctness_score": current_score.correctness_score,
                                    "error_msg": current_score.error_msg,
                                    "raw_text": raw_text,
                                }
                            )

                            # Periodic save of failed dataset
                            if len(failed_dataset) % save_every == 0:
                                fdf = pd.DataFrame(failed_dataset)
                                fdf.to_csv(
                                    os.path.join(out_dir, "failed_dataset.csv"),
                                    index=False,
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
                if t in type_successful_prompts:
                    type_successful_prompts[t].append(parsed["prompt"])

                current_count += 1
                consecutive_retries = 0
                pbar.update(1)

                # Periodic save
                if len(dataset) % save_every == 0:
                    df = pd.DataFrame(dataset)
                    df.to_csv(os.path.join(out_dir, "dataset.csv"), index=False)
                    print(f"  [INFO] Saved progress ({len(dataset)} samples)")

            except Exception as e:
                print(f"Error generating sample for {t}: {e}")
                break

        pbar.close()

        if fatal_error:
            break

    # Save final results
    df = pd.DataFrame(dataset)
    df.to_csv(os.path.join(out_dir, "dataset.csv"), index=False)

    if failed_dataset:
        fdf = pd.DataFrame(failed_dataset)
        fdf.to_csv(os.path.join(out_dir, "failed_dataset.csv"), index=False)

    print(f"Dataset saved to {out_dir}")
