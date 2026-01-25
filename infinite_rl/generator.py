import json
import random
from typing import Dict, List, Any
from .prompts import SYNTHESIS_SYSTEM_PROMPT, TASK_SYSTEM_PROMPTS, TYPE_PROMPTS


class DatasetGenerator:
    def __init__(self):
        self.puzzles = self.load_puzzles()

    def load_puzzles(self) -> Dict[str, Any]:
        """Load available puzzles. For now, hardcoded."""
        return {
            "QuadraticRoot": {
                "language": "python",
                "examples": [
                    {"coeffs": [1.0, -3.0, 2.0]},  # x^2 - 3x + 2 = 0, root at 1
                    {"coeffs": [2.5, 1.3, -0.5]},
                ],
            },
            "AllQuadraticRoots": {
                "language": "python",
                "examples": [
                    {"coeffs": [-3.0, 2.0]},  # x^2 - 3x + 2 = 0, roots 1,2
                    {"coeffs": [1.3, -0.5]},
                ],
            },
            "SumOfDigits": {
                "language": "python",
                "examples": [
                    {"s": 10},
                    {"s": 5},
                ],
            },
        }

    def generate_puzzle_sample(self, puzzle_name: str) -> Dict[str, Any]:
        """Generate a sample for a puzzle."""
        puzzle = self.puzzles[puzzle_name]
        inputs = random.choice(puzzle["examples"])

        prompt = f"Implement a sol function that solves the {puzzle_name} puzzle with inputs {inputs}."

        # Get the correct answer using the sol function
        if puzzle_name == "QuadraticRoot":
            coeffs = inputs["coeffs"]
            a, b, c = coeffs
            answer = (-b + (b**2 - 4 * a * c) ** 0.5) / (2 * a)
        elif puzzle_name == "AllQuadraticRoots":
            coeffs = inputs["coeffs"]
            b, c = coeffs
            delta = (b**2 - 4 * c) ** 0.5
            answer = [(-b + delta) / 2, (-b - delta) / 2]
        elif puzzle_name == "SumOfDigits":
            s = inputs["s"]
            # Simple solution: use as many 9s as possible
            nines = s // 9
            remainder = s % 9
            answer = "9" * nines + str(remainder) if remainder else "9" * nines

        response = f"""[PROMPT]
{prompt}

[ANSWER]
{json.dumps(answer)}

[RESPONSE]
To solve this puzzle, implement the sol function as follows:
<answer>
```python
def sol(**inputs):
    # Implementation here
    pass
```
</answer>
"""

        return {
            "prompt": prompt,
            "answer": json.dumps(
                {
                    "puzzle": puzzle_name,
                    "inputs": inputs,
                    "language": puzzle["language"],
                }
            ),
            "response": response,
            "task": "puzzle",
            "puzzle": puzzle_name,
            "inputs": inputs,
            "language": puzzle["language"],
        }

    def generate_dataset(
        self, num_samples: int, task_dist: List[float]
    ) -> List[Dict[str, Any]]:
        """Generate a dataset of samples."""
        # For now, only puzzle tasks
        samples = []
        puzzle_names = list(self.puzzles.keys())

        for _ in range(num_samples):
            puzzle_name = random.choice(puzzle_names)
            sample = self.generate_puzzle_sample(puzzle_name)
            samples.append(sample)

        return samples
