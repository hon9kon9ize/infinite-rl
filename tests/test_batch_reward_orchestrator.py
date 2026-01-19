import unittest
from infinite_rl.reward_orchestrator import BatchRewardOrchestrator
from infinite_rl.reward_functions.reward_function import RewardFunctionScore


class TestBatchRewardOrchestrator(unittest.TestCase):
    def test_compute_batch_basic(self):
        # Use a small worker pool for tests
        batcher = BatchRewardOrchestrator(
            num_workers=2,
            include_length=True,
            include_repetition=True,
            gatekeeping=True,
        )

        samples = [
            {
                "model_output": "<answer>42</answer>",
                "expected_output": 42,
                "task": "math",
                "lang": "en",
            },
            {
                "model_output": "<answer>wrong</answer>",
                "expected_output": 42,
                "task": "math",
                "lang": "en",
            },
        ]

        results = batcher.compute_batch(samples)
        self.assertEqual(len(results), 2)

        r0 = results[0]
        r1 = results[1]
        self.assertIsInstance(r0, RewardFunctionScore)
        # Math should be exact: correct answer yields correctness 1.0
        self.assertEqual(r0.correctness_score, 1.0)
        self.assertGreaterEqual(r0.aux_score, 0.0)

        self.assertIsInstance(r1, RewardFunctionScore)
        # gatekeeping should zero out the incorrect sample
        self.assertEqual(r1.format_score, 0.0)
        self.assertEqual(r1.correctness_score, 0.0)
        self.assertEqual(r1.aux_score, 0.0)


if __name__ == "__main__":
    unittest.main()
