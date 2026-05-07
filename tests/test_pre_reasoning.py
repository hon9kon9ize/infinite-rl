import json
import os
import tempfile
import unittest
from unittest.mock import MagicMock

from infinite_rl.curriculum import CurriculumLearning
from infinite_rl.dynamic_dataset import DynamicCurriculumDataset
from infinite_rl.reward_functions import RewardFunctionScore
from infinite_rl.session import Session


class TestPreReasoning(unittest.TestCase):
    def _jsonl_dataset(self):
        row = {
            "messages": [
                {"role": "system", "content": "Be concise."},
                {"role": "user", "content": "What is the capital of France?"},
                {"role": "assistant", "content": "The capital of France is Paris."},
            ],
            "lang": "en",
        }
        tmp = tempfile.NamedTemporaryFile("w", suffix=".jsonl", delete=False)
        tmp.write(json.dumps(row) + "\n")
        tmp.close()
        self.addCleanup(lambda: os.path.exists(tmp.name) and os.unlink(tmp.name))
        return tmp.name

    def test_session_loads_chat_jsonl_as_pre_reasoning(self):
        path = self._jsonl_dataset()
        session = Session(pre_reasoning_dataset=path)

        self.assertEqual(len(session.pre_reasoning_tasks), 1)
        task = session.create_pre_reasoning_task(session.pre_reasoning_tasks[0])

        self.assertEqual(task.task_type, "pre_reasoning")
        self.assertEqual(task.expected_answer["reference_answer"], "The capital of France is Paris.")
        self.assertIsInstance(task.prompt, list)
        self.assertEqual(task.prompt[-1]["role"], "user")
        self.assertIn("<think>", task.prompt[-1]["content"])
        self.assertIn("<answer>", task.prompt[-1]["content"])

    def test_pre_reasoning_dataset_reuses_prompt_for_eight_generations(self):
        path = self._jsonl_dataset()
        curriculum = CurriculumLearning(
            pre_reasoning_dataset=path,
            pre_reasoning_learning_rate=1.0,
            num_generations=8,
            use_lang_consistency=False,
        )
        dataset = DynamicCurriculumDataset(curriculum, num_samples=8)

        first = dataset[0]["task_metadata"]["task_id"]
        eighth = dataset[7]["task_metadata"]["task_id"]

        self.assertEqual(first, eighth)
        self.assertEqual(dataset[0]["task_metadata"]["task_type"], "pre_reasoning")

    def test_blank_reasoning_scores_zero_for_pre_reasoning(self):
        path = self._jsonl_dataset()
        curriculum = CurriculumLearning(
            pre_reasoning_dataset=path,
            pre_reasoning_learning_rate=1.0,
            num_generations=2,
            use_format=True,
            use_lang_consistency=False,
            use_reasoning_steps=False,
            use_response_content=False,
            use_length=False,
            use_llm_judge=True,
            llm_judge_kwargs={
                "api_host": "localhost",
                "api_port": 8000,
                "model_name": "judge",
            },
            aux_weight=0.5,
        )
        task = curriculum.get_prompt()
        curriculum.aux_reward_functions["llm_judge"].compute_rewards_batch = MagicMock(
            return_value=[
                [
                    RewardFunctionScore(score=0.2, info="bad"),
                    RewardFunctionScore(score=0.9, info="good"),
                ]
            ]
        )

        blank = "<think>blank</think><answer>The capital of France is Paris.</answer>"
        valid_reasoning = (
            "<think>"
            "The user asks for a factual capital city. France is a country in "
            "Europe, and its internationally recognized capital city is Paris, "
            "so the final response should state that directly."
            "</think><answer>The capital of France is Paris.</answer>"
        )
        scores = curriculum.compute_rewards(task.task_id, [blank, valid_reasoning])

        self.assertEqual(scores[0], 0.0)
        self.assertGreater(scores[1], scores[0])
        self.assertEqual(task.generations[0].primary_score, 0.2)
        self.assertEqual(task.generations[1].primary_score, 0.9)
        self.assertEqual(curriculum.success_windows, {})

    def test_pre_reasoning_requires_llm_judge(self):
        path = self._jsonl_dataset()
        curriculum = CurriculumLearning(
            pre_reasoning_dataset=path,
            pre_reasoning_learning_rate=1.0,
            num_generations=1,
            use_llm_judge=False,
        )
        task = curriculum.get_prompt()

        with self.assertRaisesRegex(ValueError, "pre_reasoning tasks require llm_judge"):
            curriculum.compute_reward(
                task.task_id,
                "<think>Reasoning with enough detail to be non-empty.</think><answer>Paris.</answer>",
            )


if __name__ == "__main__":
    unittest.main()
