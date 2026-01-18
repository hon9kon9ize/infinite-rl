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

        # Monkeypatch the executor.run_single to return a high similarity
        original_run = reward_fn.executor.run_single

        def fake_run_single(code, lang):
            # Expect (document, query) tuple and lang 'qwen3_embed'
            self.assertEqual(lang, "qwen3_embed")
            # Return high similarity as string
            return "0.95", ""

        reward_fn.executor.run_single = fake_run_single

        score = reward_fn.compute_reward(example["response"], example["answer"])
        self.assertEqual(score.format_score, 1.0)
        self.assertAlmostEqual(score.correctness_score, 0.95, places=3)

        # Restore
        reward_fn.executor.run_single = original_run

    def test_executor_error_propagation(self):
        reward_fn = SummarizationRewardFunction(task_name="summarization")
        reward_fn.initialize()

        def error_run_single(code, lang):
            return None, "Executor Error: no qwen3"

        reward_fn.executor.run_single = error_run_single

        score = reward_fn.compute_reward("<answer>summary</answer>", "ref summary")
        self.assertEqual(score.format_score, 1.0)
        self.assertEqual(score.correctness_score, 0.0)
        self.assertIn("Executor Error", score.error_msg)
