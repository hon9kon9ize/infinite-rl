import unittest
from pathlib import Path
from infinite_rl.parser import ExampleParser
from infinite_rl.reward_functions.summarization import SummarizationRewardFunction


class TestSummarizationRewardFunction(unittest.TestCase):
    def setUp(self):
        package_dir = Path(__file__).parent.parent / "infinite_rl" / "examples"
        if not package_dir.exists():
            package_dir = Path(__file__).parent.parent / "examples"
        self.examples = ExampleParser.get_all_examples(package_dir)

    def test_summarization_example_with_executor(self):
        example = self.examples.get("SUMMARIZATION")
        self.assertIsNotNone(example)

        # Patch the executor to avoid requiring qwen3 runtime in CI
        reward_fn = SummarizationRewardFunction(task_name="summarization")
        reward_fn.initialize()

        # Monkeypatch the executor.get_embedding to return deterministic embeddings
        original_get = reward_fn.executor.get_embedding

        def fake_get_embedding(text, role="document"):
            # If called for document return vector [1,0,0], for query [0.95, 0.1, 0]
            if role == "document":
                return [1.0, 0.0, 0.0], ""
            return [0.95, 0.1, 0.0], ""

        reward_fn.executor.get_embedding = fake_get_embedding

        score = reward_fn.compute_reward(example["response"], example["answer"])
        self.assertEqual(score.format_score, 1.0)
        self.assertAlmostEqual(
            score.correctness_score,
            reward_fn.executor.cosine_similarity([1.0, 0.0, 0.0], [0.95, 0.1, 0.0]),
            places=6,
        )

        # Restore
        reward_fn.executor.get_embedding = original_get

    def test_executor_error_propagation(self):
        reward_fn = SummarizationRewardFunction(task_name="summarization")
        reward_fn.initialize()

        def error_get_embedding(text, role="document"):
            return None, "Executor Error: no qwen3"

        reward_fn.executor.get_embedding = error_get_embedding

        score = reward_fn.compute_reward("<answer>summary</answer>", "ref summary")
        self.assertEqual(score.format_score, 1.0)
        self.assertEqual(score.correctness_score, 0.0)
        self.assertIn("Executor Error", score.error_msg)
