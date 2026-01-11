import os
import sys
import random

# Mocking parts of the system to simulate prompt generation
from infinite_rl.prompts import SYSTEM_PROMPT, TYPE_PROMPTS, RECTIFY_PROMPT


def dry_run_prompt(task_type, successful_prompts):
    type_prompt_base = TYPE_PROMPTS.get(task_type, f"Generate a sample for {task_type}")

    seed_context = ""
    # Use the same logic as in generator.py
    if successful_prompts:
        random_seed = random.choice(successful_prompts)
        seed_context = f'\n\nHere is a prompt we previously generated: "{random_seed}". Please generate something DIFFERENT and more complex than this.'

    full_query = f"Generate a new {task_type} example. {type_prompt_base}{seed_context}"

    print("=" * 50)
    print(f"DRY RUN: PROMPT FOR TASK TYPE '{task_type}'")
    print("=" * 50)
    print(f"SYSTEM INSTRUCTION:\n{SYSTEM_PROMPT}")
    print("-" * 50)
    print(f"USER QUERY:\n{full_query}")
    print("=" * 50)


# Simulate the problem: HTML task with a Python seed
python_seed = "Write a Python function called `fibonacci` that takes an integer `n` as input and returns the nth Fibonacci number."
print("\n[SIMULATING THE BUG: HTML task pulling a Python seed]")
dry_run_prompt("html", [python_seed])

# Simulate the fix: HTML task with an HTML seed
html_seed = (
    "Create a responsive 3-column layout using CSS Flexbox with a header and footer."
)
print("\n[SIMULATING THE FIX: HTML task pulling only HTML seeds]")
dry_run_prompt("html", [html_seed])
