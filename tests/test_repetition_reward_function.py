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
        # No repetition should return perfect score (1.0)
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
        # High repetition -> lower score (not perfect 1.0)
        self.assertLess(s.score, 1.0)
        self.assertGreaterEqual(s.score, 0.0)
        # Verify info message is present
        self.assertIn("Repetition detected", s.info)

    def test_missing_tag_returns_zero(self):
        task = Task(
            task_id="test_3",
            task_name="test",
            task_type="puzzle",
            level=1,
            prompt="Test",
            expected_answer="answer",
            language="en",
            model_output="no think tag here",
        )
        s = self.fn.compute_reward(task)
        # Missing tag should return 0.0
        self.assertEqual(s.score, 0.0)
        self.assertIn("No content found", s.info)

    def test_empty_output_returns_zero(self):
        task = Task(
            task_id="test_4",
            task_name="test",
            task_type="puzzle",
            level=1,
            prompt="Test",
            expected_answer="answer",
            language="en",
            model_output="",
        )
        s = self.fn.compute_reward(task)
        # Empty output should return 0.0
        self.assertEqual(s.score, 0.0)


if __name__ == "__main__":
    unittest.main()
