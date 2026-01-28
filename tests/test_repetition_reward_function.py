import unittest
from infinite_rl.reward_functions.repetition import RepetitionRewardFunction
from infinite_rl.task import Task


class TestRepetitionRewardFunction(unittest.TestCase):
    def setUp(self):
        self.fn = RepetitionRewardFunction(target_tag="think")
        self.fn.initialize()

    def test_no_repetition_full_score(self):
        task = Task(
            task_id="test_1",
            task_name="test",
            task_type="puzzle",
            level=1,
            prompt="Test",
            expected_answer="answer",
            language="en",
            model_output="<think>hello world</think>",
        )
        s = self.fn.compute_reward(task)
        # Aux-only reward: returned in unified `score` field
        self.assertAlmostEqual(s.score, 1.0, places=5)

    def test_repetition_penalized(self):
        task = Task(
            task_id="test_2",
            task_name="test",
            task_type="puzzle",
            level=1,
            prompt="Test",
            expected_answer="answer",
            language="en",
            model_output="<think>hello hello hello hello</think>",
        )
        s = self.fn.compute_reward(task)
        # High repetition -> lower score than 1.0
        self.assertLess(s.score, 1.0)
        self.assertGreaterEqual(s.score, 0.0)


if __name__ == "__main__":
    unittest.main()
