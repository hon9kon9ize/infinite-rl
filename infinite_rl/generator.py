import os
import time
import random
import threading
from concurrent.futures import ThreadPoolExecutor
from .prompts import (
    SYNTHESIS_SYSTEM_PROMPT,
    TYPE_PROMPTS,
    RECTIFY_PROMPT,
    TASK_SYSTEM_PROMPTS,
)
from .parser import ExampleParser
from .reward_functions.coding import PythonRewardFunction, JavascriptRewardFunction
from .reward_functions.math import MathRewardFunction
from .reward_functions.lang_consistency import LangConsistencyRewardFunction


def get_reward_function(task_type, timeout=5):
    """Retrieve the appropriate reward function for a task type."""
    if task_type == "python":
        return PythonRewardFunction(task_name=task_type, timeout=timeout)
    elif task_type == "javascript":
        return JavascriptRewardFunction(task_name=task_type, timeout=timeout)
    elif task_type == "math":
        return MathRewardFunction(task_name="math", timeout=timeout)
    elif task_type == "lang_consistency":
        return LangConsistencyRewardFunction(
            task_name="lang_consistency", timeout=timeout
        )
    return None


def log_failed_sample(
    failed_list, out_dir, task_type, parsed, score, raw_text, rectified
):
    """Log a failed sample to the failed dataset and save immediately.

    This function now computes a separate format score using the FormatRewardFunction
    instantiated for the specific `task_type` so format is recorded independently.
    """
    # Compute format score using the format reward for this task type
    try:
        from .reward_functions.format import FormatRewardFunction

        fmt_fn = FormatRewardFunction(task_name=task_type)
        fmt_score = float(fmt_fn.compute_reward(parsed.get("response", ""), None).score)
    except Exception:
        fmt_score = 0.0

    entry = {
        "type": task_type,
        "prompt": parsed.get("prompt", ""),
        "answer": parsed.get("answer", ""),
        "response": parsed.get("response", ""),
        "format_score": fmt_score,
        "correctness_score": float(getattr(score, "score", 0.0)),
        "error_msg": getattr(score, "error_msg", ""),
        "raw_text": raw_text,
        "rectified": rectified,
    }
    failed_list.append(entry)

    # Save to CSV immediately
    save_path = os.path.join(out_dir, "failed_dataset.csv")
    try:
        import pandas as pd

        # Check if file exists to decide on header
        file_exists = os.path.isfile(save_path)
        # We append just the new entry to the file for efficiency and to avoid overwriting previous runs
        df_new = pd.DataFrame([entry])
        df_new.to_csv(save_path, mode="a", index=False, header=not file_exists)
        print(f"  [DEBUG] Logged failure to {os.path.abspath(save_path)}")
    except Exception as e:
        print(f"  [ERROR] Failed to save {save_path}: {e}")


def generate_dataset(
    model_name,
    num_samples,
    out_dir,
    save_every=10,
    max_retries=5,
    timeout=5,
    task_dist="0.5,0.5",
    debug=False,
    num_threads=1,
):
    # Lazy import genai to avoid import errors when using as library
    from google import genai
    from google.genai import types

    # Create debug directory if it doesn't exist and debug is enabled
    debug_dir = os.path.join(out_dir, "debug_prompts")
    if debug and not os.path.exists(debug_dir):
        os.makedirs(debug_dir)

    # Parse task distribution: [code, math]
    try:
        dist_values = [float(v) for v in task_dist.split(",")]
        if len(dist_values) != 2:
            raise ValueError("task_dist must contain exactly 2 values")
    except Exception as e:
        print(f"Error parsing task_dist: {e}. Using default.")
        dist_values = [0.5, 0.5]

    # Split coding weight across python and javascript equally for sub-task granularity
    coding_weight = dist_values[0]
    distribution = {
        "python": coding_weight / 2.0,
        "javascript": coding_weight / 2.0,
        "math": dist_values[1],
    }

    # Resume Logic: Check for existing dataset
    dataset_path = os.path.join(out_dir, "dataset.csv")
    existing_counts = {}
    dataset = []

    if os.path.exists(dataset_path):
        try:
            existing_df = pd.read_csv(dataset_path)
            # Remove any rows with missing types
            if "type" in existing_df.columns:
                existing_df = existing_df.dropna(subset=["type"])
                dataset = existing_df.to_dict("records")
                existing_counts = existing_df["type"].value_counts().to_dict()
                print(
                    f"Resuming from existing dataset at {dataset_path} ({len(dataset)} samples found)"
                )
        except Exception as e:
            print(f"Warning: Could not load existing dataset: {e}. Starting fresh.")
            dataset = []

    # Calculate target counts based on the TOTAL requested samples
    target_counts = {t: int(num_samples * weight) for t, weight in distribution.items()}

    # Ensure we reach total samples by adding remainder to the main type (coding)
    total_calculated = sum(target_counts.values())
    if total_calculated < num_samples:
        # Find the type with non-zero weight to add remainder, default to coding
        active_types = [t for t, w in distribution.items() if w > 0]
        addition_type = active_types[0] if active_types else "python"
        target_counts[addition_type] += num_samples - total_calculated

    # Calculate how many MORE we need for each type to reach targets
    type_counts = {}
    for t in target_counts:
        needed = max(0, target_counts[t] - existing_counts.get(t, 0))
        type_counts[t] = needed

    actual_num_to_generate = sum(type_counts.values())

    print(f"Target total: {num_samples} samples.")
    print(f"Already existing: {len(dataset)} samples ({existing_counts})")
    print(f"Remaining to generate: {actual_num_to_generate} samples.")
    print(f"Remaining Distribution: {type_counts}")

    failed_dataset = []
    # Distribution includes sub-types for coding
    type_successful_prompts = {
        "python": [],
        "javascript": [],
        "math": [],
    }

    # Load initial seed prompts from examples directory for diversity
    examples_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "examples")
    if os.path.exists(examples_dir):
        # We'll map filenames to task types
        file_to_type = {
            "PYTHON": "python",
            "JAVASCRIPT": "javascript",
            "MATH": "math",
            "LANG_CONSISTENCY": "lang_consistency",
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
                            if task_t in [
                                "python",
                                "javascript",
                            ]:
                                type_successful_prompts["python"].append(
                                    parsed["prompt"]
                                )
                    except Exception as e:
                        print(f"Warning: Could not load seed example {filename}: {e}")

    # Also load prompts from the existing dataset to avoid duplicates if possible
    # and provide more context for diversity.
    for sample in dataset:
        t = sample.get("type")
        p = sample.get("prompt")
        if t and p and isinstance(p, str):
            if t in type_successful_prompts:
                type_successful_prompts[t].append(p)

    # Configure Gemini
    client = genai.Client()

    # Threading support
    dataset_lock = threading.Lock()
    failed_dataset_lock = threading.Lock()
    success_prompts_lock = threading.Lock()
    save_lock = threading.Lock()

    # Pre-initialize reward functions (some might load models)
    reward_functions = {}
    for t in type_counts.keys():
        if type_counts[t] > 0:
            rf = get_reward_function(t, timeout=timeout)
            if rf:
                rf.initialize()
                reward_functions[t] = rf

    def generate_single_sample(t, pbar):
        type_prompt_base = TYPE_PROMPTS.get(t, f"Generate a sample for {t}")
        system_inst = TASK_SYSTEM_PROMPTS.get(t, SYNTHESIS_SYSTEM_PROMPT)
        reward_fn = reward_functions.get(t)

        attempt = 0
        while attempt < max_retries:
            attempt += 1
            if attempt > 1:
                wait_time = min(60, 2**attempt)
                print(
                    f"  [DEBUG] Retrying task {t} (attempt {attempt}/{max_retries}) in {wait_time}s..."
                )
                time.sleep(wait_time)

            try:
                # Diversity: Pick a random seed prompt
                seed_context = ""
                with success_prompts_lock:
                    current_seeds = type_successful_prompts.get(t, [])
                    if current_seeds:
                        random_seed = random.choice(current_seeds)
                        seed_context = f'\n\nHere is a prompt we previously generated: "{random_seed}". Please generate something DIFFERENT and more complex than this.'

                full_query = (
                    f"Generate a new {t} example. {type_prompt_base}{seed_context}"
                )

                # Log the prompt for debugging
                if debug:
                    with dataset_lock:  # simple way to avoid filename collision
                        dbg_id = len(dataset) + len(failed_dataset)
                    debug_file = os.path.join(debug_dir, f"{t}_{dbg_id}.txt")
                    with open(debug_file, "w") as f:
                        f.write(f"--- SYSTEM PROMPT ---\n{system_inst}\n\n")
                        f.write(f"--- USER PROMPT (Task: {t}) ---\n{full_query}\n")

                response = client.models.generate_content(
                    model=model_name,
                    contents=full_query,
                    config=types.GenerateContentConfig(
                        system_instruction=system_inst,
                    ),
                )
                raw_text = response.text
                parsed = ExampleParser.parse_text(raw_text)

                # Validation
                quality_score = 1.0
                if reward_fn:
                    score = reward_fn.compute_reward(
                        parsed["response"], parsed["answer"]
                    )

                    # Compute format score using a task-specific FormatRewardFunction
                    try:
                        from .reward_functions.format import FormatRewardFunction

                        fmt_fn = FormatRewardFunction(task_name=t)
                        fmt_score = float(
                            fmt_fn.compute_reward(
                                parsed.get("response", ""), None
                            ).score
                        )
                    except Exception:
                        fmt_score = 0.0

                    quality_score = (
                        fmt_score + float(getattr(score, "score", 0.0))
                    ) / 2.0

                    if quality_score < 0.8:
                        # Log initial failure
                        with failed_dataset_lock:
                            log_failed_sample(
                                failed_dataset,
                                out_dir,
                                t,
                                parsed,
                                score,
                                raw_text,
                                False,
                            )

                        # Attempt Rectification
                        rectify_count = 0
                        max_rectify = 2
                        current_raw = raw_text
                        current_score = score

                        while rectify_count < max_rectify and (quality_score < 0.8):
                            rectify_count += 1
                            current_fmt = fmt_score
                            error_info = f"Format Score: {current_fmt:.2f}, Correctness Score: {float(getattr(current_score, 'score', 0.0)):.2f}"
                            if getattr(current_score, "error_msg", None):
                                error_info += f"\nError details: {current_score.error_msg.values()}"

                            rectify_query = RECTIFY_PROMPT.format(
                                error_info=error_info, current_raw=current_raw
                            )

                            try:
                                response = client.models.generate_content(
                                    model=model_name,
                                    contents=rectify_query,
                                    config=types.GenerateContentConfig(
                                        system_instruction=system_inst,
                                    ),
                                )
                                current_raw = response.text
                                parsed = ExampleParser.parse_text(current_raw)

                                # Re-evaluate

                                current_score = reward_fn.compute_reward(
                                    parsed["response"],
                                    parsed["answer"],
                                )
                                try:
                                    fmt_score = float(
                                        fmt_fn.compute_reward(
                                            parsed.get("response", ""), None
                                        ).score
                                    )
                                except Exception:
                                    fmt_score = 0.0

                                quality_score = (
                                    fmt_score
                                    + float(getattr(current_score, "score", 0.0))
                                ) / 2.0
                                raw_text = current_raw

                                if quality_score < 0.8:
                                    with failed_dataset_lock:
                                        log_failed_sample(
                                            failed_dataset,
                                            out_dir,
                                            t,
                                            parsed,
                                            current_score,
                                            current_raw,
                                            True,
                                        )
                            except Exception:
                                break

                        if quality_score < 0.8:
                            continue  # Retry with a new prompt

                # Success
                sample = {
                    "type": t,
                    "prompt": parsed["prompt"],
                    "answer": parsed["answer"],
                    "response": parsed["response"],
                    "quality_score": quality_score,
                }
                with dataset_lock:
                    dataset.append(sample)
                    ds_len = len(dataset)

                with success_prompts_lock:
                    if t in type_successful_prompts:
                        type_successful_prompts[t].append(parsed["prompt"])

                pbar.update(1)

                # Periodic save
                if ds_len % save_every == 0:
                    with save_lock:
                        df = pd.DataFrame(dataset)
                        df.to_csv(os.path.join(out_dir, "dataset.csv"), index=False)
                        pbar.write(f"  [INFO] Saved progress ({ds_len} samples)")

                return True  # Sample generated successfully

            except Exception as e:
                print(f"Error generating sample for {t}: {e}")
                continue

        return False  # Failed after max_retries

    # Flatten tasks to run them in parallel
    all_tasks = []
    for t, count in type_counts.items():
        for _ in range(count):
            all_tasks.append(t)
    random.shuffle(all_tasks)

    from tqdm import tqdm

    pbar = tqdm(
        total=max(num_samples, len(dataset)),
        initial=len(dataset),
        desc="Total Progress",
    )

    with ThreadPoolExecutor(max_workers=num_threads) as executor:
        list(executor.map(lambda t: generate_single_sample(t, pbar), all_tasks))

    pbar.close()

    # Save final results
    with save_lock:
        df = pd.DataFrame(dataset)
        df.to_csv(os.path.join(out_dir, "dataset.csv"), index=False)
    print(f"Dataset saved to {out_dir}")
