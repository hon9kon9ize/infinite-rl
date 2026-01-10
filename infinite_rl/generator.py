import os
import pandas as pd
import google.generativeai as genai
import json
from dotenv import load_dotenv
from .prompts import SYSTEM_PROMPT, TYPE_PROMPTS

load_dotenv()


def generate_dataset(model_name, types, num_samples, out_dir):
    print(f"Generating {num_samples} samples of types {types} using {model_name}...")

    samples_per_type = num_samples // len(types)
    remainder = num_samples % len(types)

    dataset = []

    # Configure Gemini
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise ValueError("GOOGLE_API_KEY not found in environment variables")

    genai.configure(api_key=api_key)
    model = genai.GenerativeModel(
        model_name=model_name, system_instruction=SYSTEM_PROMPT
    )

    for i, t in enumerate(types):
        count = samples_per_type + (1 if i < remainder else 0)
        print(f"Generating {count} samples for type: {t}")

        type_prompt = TYPE_PROMPTS.get(t, f"Generate a sample for {t}")

        for _ in range(count):
            try:
                response = model.generate_content(
                    type_prompt,
                    generation_config={"response_mime_type": "application/json"},
                )
                data = json.loads(response.text)
                sample = {
                    "type": t,
                    "prompt": data.get("prompt"),
                    "chosen": data.get("chosen"),
                    "rejected": data.get("rejected"),
                }
                dataset.append(sample)
            except Exception as e:
                print(f"Error generating sample for {t}: {e}")

    # Save dataset.csv
    df = pd.DataFrame(dataset)
    df.to_csv(os.path.join(out_dir, "dataset.csv"), index=False)

    # Save reward function
    reward_fn_path = os.path.join(out_dir, "reward_function.py")
    with open(reward_fn_path, "w") as f:
        f.write("def reward_function(prompt, response, task_type):\n")
        f.write('    """\n')
        f.write(
            "    Calculates a reward score for a given response based on the prompt and task type.\n"
        )
        f.write("    Returns a float between 0.0 and 1.0.\n")
        f.write('    """\n')
        f.write("    score = 0.5\n")
        f.write("    \n")
        f.write("    if task_type == 'coding':\n")
        f.write("        # Example: Check for code blocks or specific keywords\n")
        f.write("        if '```' in response:\n")
        f.write("            score += 0.2\n")
        f.write("        if 'def ' in response or 'import ' in response:\n")
        f.write("            score += 0.2\n")
        f.write("            \n")
        f.write("    elif task_type == 'math':\n")
        f.write("        # Example: Check for numerical values or equations\n")
        f.write("        if any(c.isdigit() for c in response):\n")
        f.write("            score += 0.2\n")
        f.write("        if '=' in response:\n")
        f.write("            score += 0.2\n")
        f.write("            \n")
        f.write("    elif task_type == 'summarization':\n")
        f.write("        # Example: Check for length relative to prompt (heuristic)\n")
        f.write("        if len(response) < len(prompt) * 0.5:\n")
        f.write("            score += 0.2\n")
        f.write("            \n")
        f.write("    elif task_type == 'creativity':\n")
        f.write(
            "        # Example: Check for descriptive language (very basic heuristic)\n"
        )
        f.write("        if len(response.split()) > 50:\n")
        f.write("            score += 0.2\n")
        f.write("            \n")
        f.write("    return min(1.0, score)\n")

    print(f"Dataset and reward function saved to {out_dir}")
