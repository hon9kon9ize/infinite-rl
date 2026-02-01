#!/usr/bin/env python3
"""Analyze the last 60 samples from curriculum_learning_log.jsonl"""

import json
from pathlib import Path
from collections import defaultdict

log_file = Path(
    "/Users/josephcheng/Projects/rl-data-geneator/tmp/curriculum_learning_log.jsonl"
)

# Read all records
records = []
with open(log_file) as f:
    for line in f:
        try:
            records.append(json.loads(line))
        except:
            pass

print(f"Total records in log: {len(records)}\n")

# Get last 60 samples
last_60 = records[-60:] if len(records) >= 60 else records

# Calculate success rate
successes = sum(1 for r in last_60 if r.get("is_correct", False))
success_rate = successes / len(last_60) if last_60 else 0

print(f"Last {len(last_60)} samples:")
print(f"  Successes: {successes}")
print(f"  Failures: {len(last_60) - successes}")
print(f"  Mean success rate: {success_rate:.4f} ({success_rate * 100:.2f}%)\n")

# Analyze by level
level_stats = defaultdict(lambda: {"total": 0, "correct": 0})
for r in last_60:
    level = r.get("level", "unknown")
    level_stats[level]["total"] += 1
    if r.get("is_correct", False):
        level_stats[level]["correct"] += 1

print("Success rate by level (last 60 samples):")
for level in sorted(level_stats.keys()):
    stats = level_stats[level]
    rate = stats["correct"] / stats["total"] if stats["total"] > 0 else 0
    print(
        f"  Level {level}: {stats['correct']}/{stats['total']} = {rate:.4f} ({rate*100:.2f}%)"
    )

# Analyze by task type
task_type_stats = defaultdict(lambda: {"total": 0, "correct": 0})
for r in last_60:
    task_type = r.get("task_type", "unknown")
    task_type_stats[task_type]["total"] += 1
    if r.get("is_correct", False):
        task_type_stats[task_type]["correct"] += 1

print("\nSuccess rate by task type (last 60 samples):")
for task_type in sorted(task_type_stats.keys()):
    stats = task_type_stats[task_type]
    rate = stats["correct"] / stats["total"] if stats["total"] > 0 else 0
    print(
        f"  {task_type}: {stats['correct']}/{stats['total']} = {rate:.4f} ({rate*100:.2f}%)"
    )

# Analyze by language
lang_stats = defaultdict(lambda: {"total": 0, "correct": 0})
for r in last_60:
    lang = r.get("language", "unknown")
    lang_stats[lang]["total"] += 1
    if r.get("is_correct", False):
        lang_stats[lang]["correct"] += 1

print("\nSuccess rate by language (last 60 samples):")
for lang in sorted(lang_stats.keys()):
    stats = lang_stats[lang]
    rate = stats["correct"] / stats["total"] if stats["total"] > 0 else 0
    print(
        f"  {lang}: {stats['correct']}/{stats['total']} = {rate:.4f} ({rate*100:.2f}%)"
    )

# Show demote threshold recommendations
print("\n" + "=" * 60)
print("DEMOTE THRESHOLD RECOMMENDATIONS:")
print("=" * 60)
print(f"Current mean success rate: {success_rate:.4f}")
print(f"\nIf you want to demote at current performance level:")
print(f"  → Set demote_threshold to {success_rate:.4f} (or slightly higher)")
print(f"\nIf current performance is ABOVE your target:")
print(f"  → Keep demote_threshold HIGHER than current rate")
print(f"  → Example: demote_threshold=0.5 would demote if rate drops below 50%")
print(f"\nIf current performance is BELOW your target:")
print(f"  → Raise demote_threshold LOWER to be stricter")
print(f"  → Example: demote_threshold=0.2 would only demote if rate drops below 20%")
