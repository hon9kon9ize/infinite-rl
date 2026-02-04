"""
Training simulator for curriculum learning.

Simulates the learning process by generating different combinations of responses
and tracking curriculum progression, reward changes, and level advancement.

Response combinations allow testing:
- think_format: whether <think> tag is present and valid
- answer_format: whether <answer> tag is present and valid
- correct: whether the answer is correct

Example usage:
    simulator = TrainingSimulator(num_generations=4, use_format=True)
    simulator.run_scenario(
        "Good Format + Correct",
        response_configs=[
            (True, True, True),   # all good
        ] * 50
    )
    simulator.print_results()
"""

import json
from typing import List, Tuple, Dict, Any
from dataclasses import dataclass, asdict
from datetime import datetime
from infinite_rl.curriculum import CurriculumLearning
from infinite_rl.task import Task


@dataclass
class RewardSnapshot:
    """Snapshot of rewards at a point in time."""

    step: int
    primary_score: float
    combined_score: float
    level: int
    success_rate: float
    is_correct: bool
    response_type: str
    task_type: str = "math"  # Type of task (math, puzzle, truthy)
    # Individual reward scores
    format_score: float = 0.0
    reasoning_steps_score: float = 0.0
    lang_consistency_score: float = 0.0
    repetition_score: float = 0.0
    length_score: float = 0.0
    llm_judge_score: float = 0.0


class TrainingSimulator:
    """Simulates curriculum learning with synthetic responses."""

    def __init__(
        self,
        num_generations: int = 4,
        use_format: bool = True,
        use_repetition: bool = False,
        use_reasoning_steps: bool = False,
        use_lang_consistency: bool = False,
        use_length: bool = False,
        warmup_step: int = 32,
        success_rate_threshold: float = 0.7,
        demote_threshold: float = 0.4,
        window_size: int = 50,
        aux_weight: float = 0.2,
        use_llm_judge: bool = False,
        llm_judge_weight: float = 0.2,
        llm_judge_kwargs: Dict[str, Any] = None,
        log_file: Optional[str] = None,
    ):
        """Initialize the simulator.

        Args:
            num_generations: Number of generations per GRPO batch
            use_format: Whether to use format validation
            use_repetition: Whether to use repetition penalty
            use_reasoning_steps: Whether to use reasoning steps reward
            use_lang_consistency: Whether to use language consistency reward
            use_length: Whether to use length regularization reward
            warmup_step: Number of warmup steps
            success_rate_threshold: Threshold for level advancement
            demote_threshold: Threshold for level demotion
            window_size: Size of sliding window for success rate tracking
            aux_weight: Weight for auxiliary rewards
            use_llm_judge: Whether to use LLM Judge for truthy tasks
            llm_judge_weight: Weight for LLM Judge reward
            llm_judge_kwargs: Additional kwargs for LLM Judge initialization
            log_file: Path to the logging file (JSON Lines format)
        """
        self.num_generations = num_generations
        self.use_format = use_format
        self.use_repetition = use_repetition
        self.use_reasoning_steps = use_reasoning_steps
        self.use_lang_consistency = use_lang_consistency
        self.use_length = use_length
        self.aux_weight = aux_weight
        self.use_llm_judge = use_llm_judge
        self.llm_judge_weight = llm_judge_weight
        self.demote_threshold = demote_threshold
        self.window_size = window_size
        self.log_file = log_file

        # Build curriculum kwargs
        curriculum_kwargs = {
            "use_format": use_format,
            "use_repetition": use_repetition,
            "use_reasoning_steps": use_reasoning_steps,
            "use_lang_consistency": use_lang_consistency,
            "use_length": use_length,
            "warmup_step": warmup_step,
            "success_rate_threshold": success_rate_threshold,
            "demote_threshold": demote_threshold,
            "window_size": window_size,
            "aux_weight": aux_weight,
            "num_generations": num_generations,
            "puzzle_one_shot": False,
            "use_llm_judge": use_llm_judge,
            "llm_judge_weight": llm_judge_weight,
            "log_file": log_file,
        }

        # Add LLM Judge kwargs if using LLM Judge
        if use_llm_judge and llm_judge_kwargs:
            curriculum_kwargs.update(llm_judge_kwargs)

        self.curriculum = CurriculumLearning(**curriculum_kwargs)
        self.snapshots: List[RewardSnapshot] = []
        self.task_counter = 0

    def generate_response(
        self,
        has_think_format: bool = True,
        has_answer_format: bool = True,
        is_correct: bool = True,
        task_type: str = "math",
    ) -> str:
        """Generate a synthetic response with specified properties.

        Args:
            has_think_format: Whether to include valid <think> tags
            has_answer_format: Whether to include valid <answer> tags
            is_correct: Whether the answer is mathematically correct
            task_type: Type of task ("math", "puzzle", "truthy")

        Returns:
            Response string with specified properties
        """
        if task_type == "puzzle":
            # For puzzles, generate JavaScript code with both think and answer tags
            think_content = (
                "<think>Let me write the solution.</think>" if has_think_format else ""
            )
            if is_correct:
                code = "function solution() { return true; }"
            else:
                code = "function solution() { return false; }"
            answer_content = f"<answer>```javascript\n{code}\n```</answer>"
            return f"{think_content}\n\n{answer_content}".strip()
        else:
            # For math/truthy, use the original format
            think_content = (
                "<think>Let me solve this step by step.</think>"
                if has_think_format
                else ""
            )
            answer = "4" if is_correct else "5"  # For "what is 2+2?" task
            answer_content = (
                f"<answer>{answer}</answer>" if has_answer_format else answer
            )

            return f"{think_content}\n\n{answer_content}".strip()

    def run_batch(
        self,
        response_configs: List[Tuple[bool, bool, bool]],
        response_type: str = "custom",
        task_type: str = "math",
    ) -> Dict[str, Any]:
        """Run a batch of responses and track statistics.

        Args:
            response_configs: List of (has_think, has_answer, is_correct) tuples
            response_type: Description of the response type
            task_type: Type of task to create ("math", "puzzle", or "truthy")

        Returns:
            Statistics for the batch
        """
        batch_results = {
            "primary_scores": [],
            "combined_scores": [],
            "is_correct_list": [],
            "response_type": response_type,
            "timestamp": datetime.now().isoformat(),
        }

        # Create a single task for this batch (representing one prompt in GRPO)
        self.task_counter += 1

        if task_type == "truthy":
            # Create truthy task
            base_task_id = f"truthy_batch_{self.task_counter}"
            task = Task(
                task_id=base_task_id,
                task_name=f"Truthy Task {self.task_counter}",
                task_type="truthy",
                level=self.curriculum.current_level,
                prompt="Which response is better?",
                expected_answer={"type": "truthy", "conversation": []},
            )
        elif task_type == "puzzle":
            # Create puzzle task (start at level 1 minimum since level 0 is math-only)
            puzzle_level = max(self.curriculum.current_level, 1)
            base_task_id = f"puzzle_batch_{self.task_counter}"
            task = Task(
                task_id=base_task_id,
                task_name=f"Puzzle Task {self.task_counter}",
                task_type="puzzle",
                level=puzzle_level,
                prompt="Solve this puzzle",
                expected_answer={
                    "puzzle": "dummy_puzzle",
                    "inputs": {},
                    "language": "javascript",
                },
            )
        else:
            # Create math task
            base_task_id = f"math_batch_{self.task_counter}"
            task = Task(
                task_id=base_task_id,
                task_name=f"Math Task {self.task_counter}",
                task_type="math",
                level=self.curriculum.current_level,
                prompt="What is 2+2?",
                expected_answer="4",
            )

        self.curriculum.session.add_task(task)

        # Process all responses for this task (GRPO batch)
        task_ids = []
        for has_think, has_answer, is_correct in response_configs:
            # Generate response
            response = self.generate_response(
                has_think, has_answer, is_correct, task_type
            )

            # Compute reward (adds generation to task)
            primary_score = self.curriculum.compute_reward(base_task_id, response)
            task_ids.append(base_task_id)

            batch_results["primary_scores"].append(primary_score)

        # Get combined rewards for all generations in this batch
        combined_scores = self.curriculum.get_rewards(task_ids)

        # Get updated task for correctness (after batch processing)
        updated_task = self.curriculum.session.get_task(base_task_id)

        # For batch results, we use the task-level correctness
        # In GRPO, success is determined at the prompt level
        batch_results["combined_scores"] = combined_scores
        batch_results["is_correct_list"] = [updated_task.is_correct] * len(
            response_configs
        )

        # Extract individual reward scores from the last generation (for snapshot)
        reward_scores = {
            "format_score": 0.0,
            "reasoning_steps_score": 0.0,
            "lang_consistency_score": 0.0,
            "repetition_score": 0.0,
            "length_score": 0.0,
            "llm_judge_score": 0.0,
        }
        if updated_task.generations:
            for reward in updated_task.latest_generation.rewards:
                if reward.reward_function_name == "format":
                    reward_scores["format_score"] = reward.score
                elif reward.reward_function_name == "reasoning_steps":
                    reward_scores["reasoning_steps_score"] = reward.score
                elif reward.reward_function_name == "lang_consistency":
                    reward_scores["lang_consistency_score"] = reward.score
                elif reward.reward_function_name == "repetition":
                    reward_scores["repetition_score"] = reward.score
                elif reward.reward_function_name == "length":
                    reward_scores["length_score"] = reward.score
                elif reward.reward_function_name == "llm_judge":
                    reward_scores["llm_judge_score"] = reward.score

        # Record snapshot for each response in the batch
        success_rate = self.curriculum.get_success_rate().get("mean_success_rate", 0.0)
        for i, (primary_score, combined_score) in enumerate(
            zip(batch_results["primary_scores"], combined_scores)
        ):
            self.snapshots.append(
                RewardSnapshot(
                    step=self.curriculum.global_step,
                    primary_score=primary_score,
                    combined_score=combined_score,
                    level=self.curriculum.current_level,
                    success_rate=success_rate,
                    is_correct=updated_task.is_correct,
                    response_type=response_type,
                    task_type=task_type,
                    format_score=reward_scores["format_score"],
                    reasoning_steps_score=reward_scores["reasoning_steps_score"],
                    lang_consistency_score=reward_scores["lang_consistency_score"],
                    repetition_score=reward_scores["repetition_score"],
                    length_score=reward_scores["length_score"],
                    llm_judge_score=reward_scores["llm_judge_score"],
                )
            )

        return batch_results

    def run_scenario(
        self,
        scenario_name: str,
        response_configs: List[Tuple[bool, bool, bool]],
        batch_size: int = 4,
        task_type: str = "math",
    ) -> Dict[str, Any]:
        """Run a complete scenario with multiple batches.

        Args:
            scenario_name: Name of the scenario
            response_configs: List of (has_think, has_answer, is_correct) tuples
            batch_size: Responses per batch (for GRPO grouping)
            task_type: Type of task to create ("math", "puzzle", or "truthy")

        Returns:
            Scenario statistics
        """
        print(f"\n{'='*70}")
        print(f"Scenario: {scenario_name}")
        print(f"Task Type: {task_type}")
        print(f"{'='*70}")
        print(f"Total responses: {len(response_configs)}")
        print(f"Batch size: {batch_size}")
        print()

        batches = []
        for i in range(0, len(response_configs), batch_size):
            batch = response_configs[i : i + batch_size]
            batch_result = self.run_batch(
                batch, response_type=scenario_name, task_type=task_type
            )
            batches.append(batch_result)

            # Print progress
            if (i // batch_size + 1) % 5 == 0:
                success_rate = self.curriculum.get_success_rate().get(
                    "mean_success_rate", 0.0
                )
                print(
                    f"  Batch {i // batch_size + 1}: "
                    f"Level={self.curriculum.current_level}, "
                    f"Step={self.curriculum.global_step}, "
                    f"Success Rate={success_rate:.1%}"
                )

        return {
            "scenario_name": scenario_name,
            "task_type": task_type,
            "batches": batches,
            "final_level": self.curriculum.current_level,
            "final_step": self.curriculum.global_step,
            "final_success_rate": self.curriculum.get_success_rate().get(
                "mean_success_rate", 0.0
            ),
        }

    def print_results(self):
        """Print formatted results of all scenarios."""
        if not self.snapshots:
            print("No snapshots recorded. Run scenarios first.")
            return

        print(f"\n{'='*70}")
        print("TRAINING SUMMARY")
        print(f"{'='*70}\n")

        # Summary statistics
        final_snapshot = self.snapshots[-1]
        print(f"Final Status:")
        print(f"  Level: {final_snapshot.level}")
        print(f"  Total Steps: {final_snapshot.step}")
        print(f"  Success Rate: {final_snapshot.success_rate:.1%}")
        print()

        # Timeline of level changes
        current_level = self.snapshots[0].level
        level_changes = [(0, current_level)]

        for snapshot in self.snapshots[1:]:
            if snapshot.level != current_level:
                level_changes.append((snapshot.step, snapshot.level))
                current_level = snapshot.level

        print("Level Changes:")
        for step, level in level_changes:
            print(f"  Step {step}: → Level {level}")
        print()

        # Response type statistics
        response_types = {}
        for snapshot in self.snapshots:
            if snapshot.response_type not in response_types:
                response_types[snapshot.response_type] = {
                    "count": 0,
                    "avg_primary": 0,
                    "avg_combined": 0,
                    "correct_count": 0,
                }

            stats = response_types[snapshot.response_type]
            stats["count"] += 1
            stats["avg_primary"] += snapshot.primary_score
            stats["avg_combined"] += snapshot.combined_score
            if snapshot.is_correct:
                stats["correct_count"] += 1

        # Normalize averages
        for response_type, stats in response_types.items():
            stats["avg_primary"] /= stats["count"]
            stats["avg_combined"] /= stats["count"]
            stats["accuracy"] = stats["correct_count"] / stats["count"]

        print("Response Type Statistics:")
        for response_type, stats in response_types.items():
            print(
                f"  {response_type}:"
                f" Count={stats['count']}, "
                f"Accuracy={stats['accuracy']:.1%}, "
                f"AvgPrimary={stats['avg_primary']:.2f}, "
                f"AvgCombined={stats['avg_combined']:.2f}"
            )
        print()

    def get_snapshots_as_json(self) -> str:
        """Get all snapshots as JSON for detailed analysis."""
        return json.dumps([asdict(s) for s in self.snapshots], indent=2, default=str)

    def save_results(self, filename: str = "training_results.json"):
        """Save detailed results to JSON file."""
        results = {
            "timestamp": datetime.now().isoformat(),
            "final_level": self.curriculum.current_level,
            "final_step": self.curriculum.global_step,
            "final_success_rate": self.curriculum.get_success_rate().get(
                "mean_success_rate", 0.0
            ),
            "snapshots": [asdict(s) for s in self.snapshots],
        }

        print(f"\nResults saved to {filename}")

    def get_results(self) -> Dict[str, Any]:
        """Get final results summary with statistics.

        Returns:
            Dictionary with final_level, final_step, final_success_rate, and stats
        """
        # Calculate average scores from snapshots
        if not self.snapshots:
            return {
                "final_level": self.curriculum.current_level,
                "final_step": self.curriculum.global_step,
                "final_success_rate": 0.0,
                "stats": {
                    "avg_primary": 0.0,
                    "avg_combined": 0.0,
                    "total_responses": 0,
                },
            }

        total_primary = sum(s.primary_score for s in self.snapshots)
        total_combined = sum(s.combined_score for s in self.snapshots)
        count = len(self.snapshots)

        return {
            "final_level": self.curriculum.current_level,
            "final_step": self.curriculum.global_step,
            "final_success_rate": self.curriculum.get_success_rate().get(
                "mean_success_rate", 0.0
            ),
            "stats": {
                "avg_primary": total_primary / count,
                "avg_combined": total_combined / count,
                "total_responses": count,
            },
        }


def main():
    """Run example training scenarios."""
    simulator = TrainingSimulator(
        num_generations=4,
        use_format=True,
        success_rate_threshold=0.7,
    )

    # Scenario 1: Perfect responses (all good)
    simulator.run_scenario(
        "Perfect Format + Correct Answers",
        response_configs=[(True, True, True)] * 50,
    )

    # Scenario 2: Good format but some wrong answers
    simulator.run_scenario(
        "Good Format + Mixed Correctness",
        response_configs=[(True, True, True)] * 35 + [(True, True, False)] * 15,
    )

    # Scenario 3: Missing think tag
    simulator.run_scenario(
        "Missing Think Tag",
        response_configs=[(False, True, True)] * 30 + [(False, True, False)] * 20,
    )

    # Scenario 4: Recovery - start bad, improve
    simulator.run_scenario(
        "Start Bad, Recovery",
        response_configs=[
            (False, True, False),  # bad start
        ]
        * 20
        + [
            (True, True, True),  # recovery
        ]
        * 30,
    )

    # Print results
    simulator.print_results()

    # Save results
    simulator.save_results("training_simulation_results.json")


if __name__ == "__main__":
    main()
