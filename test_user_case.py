#!/usr/bin/env python
"""Test the actual user-reported case."""

from infinite_rl.reward_functions.format import FormatRewardFunction
from infinite_rl.task import Task

# User's actual response
user_response = """first think about the total slices for small pizzas:
4 small pizzas * 6 slices per small pizza = 24 slices

next, calculate the slices for medium pizzas:
5 medium pizzas * 8 slices per medium pizza = 40 slices

now add these together:
24 slices + 40 slices = 64 slices

There are 15 pizzas.
4 small and 5 medium pizzas left:
15 - 4 - 5 = 6 large pizzas

now, calculate slices on large pizzas:
6 large pizzas * 12 slices per large pizza = 72 slices

add the small and medium slices:
64 slices + 72 slices = 136

<think>
first, calculate the slices for small pizzas:
4 small pizzas * 6 slices per small pizza = 24 slices

next, calculate the slices for medium pizzas:
5 medium pizzas * 8 slices per medium pizza = 40 slices

now add these together:
24 slices + 40 slices = 64 slices

third, calculate the slices for large pizzas:
6 large pizzas * 12 slices per large pizza = 72 slices

fourth, add these together:
64 slices + 72 slices = 136

</think>

<answer>136</answer>"""

# Test format_think (should fail because content before <think>)
fn_think = FormatRewardFunction(task_name="format_think", target_tag="think")
fn_think.initialize()

task_think = Task(
    task_id="test_user_case_think",
    task_name="test",
    task_type="math",
    level=1,
    prompt="Test",
    expected_answer="136",
    language="en",
    model_output=user_response,
)

score_think = fn_think.compute_reward(task_think)
print(f"Format Think Score: {score_think.score}")
print(f"Format Think Info: {score_think.info}")
print()

# Test format_answer (should pass because <answer> is after <think>)
fn_answer = FormatRewardFunction(task_name="format_answer", target_tag="answer")
fn_answer.initialize()

task_answer = Task(
    task_id="test_user_case_answer",
    task_name="test",
    task_type="math",
    level=1,
    prompt="Test",
    expected_answer="136",
    language="en",
    model_output=user_response,
)

score_answer = fn_answer.compute_reward(task_answer)
print(f"Format Answer Score: {score_answer.score}")
print(f"Format Answer Info: {score_answer.info}")
print()

# Summary
print("=" * 60)
if score_think.score == -1.0:
    print("✅ FIX WORKING: Think tag correctly penalized for content before tag")
else:
    print(f"❌ BUG STILL EXISTS: Think score = {score_think.score} (expected -1.0)")

if score_answer.score == -1.0:
    print("✅ CORRECT: Answer tag also penalized (content before it too)")
else:
    print(f"⚠️  Answer tag score = {score_answer.score}")
