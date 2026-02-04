"""Unit tests for LLMJudgeRewardFunction."""

import unittest
from unittest.mock import Mock, patch, MagicMock
import requests
from infinite_rl.reward_functions.llm_judge import LLMJudgeRewardFunction
from infinite_rl.task import Task


class MockTokenizer:
    """Mock tokenizer for testing without transformers dependency."""

    def __init__(self):
        self.bos_token = "<BOS>"

    def apply_chat_template(self, conversation, tokenize=False):
        """Mock chat template application.

        Formats conversation as simple concatenation without actual tokenization.
        """
        parts = []
        for msg in conversation:
            role = msg.get("role", "").upper()
            content = msg.get("content", "")
            parts.append(f"[{role}] {content}")

        result = " ".join(parts)
        # Add BOS token prefix (will be removed by apply_chat_template caller)
        return self.bos_token + result


class TestLLMJudgeRewardFunction(unittest.TestCase):
    """Test suite for LLMJudgeRewardFunction."""

    def setUp(self):
        """Set up test fixtures."""
        self.api_host = "localhost"
        self.api_port = 8000
        self.model_name = "Skywork/Skywork-Reward-V2-Qwen3-4B"

    def create_reward_function(self, **kwargs):
        """Helper to create reward function with test defaults.

        Note: Does NOT initialize automatically, allowing tests to control
        initialization with mocks or manually set up tokenizer.
        """
        params = {
            "api_host": self.api_host,
            "api_port": self.api_port,
            "model_name": self.model_name,
        }
        params.update(kwargs)
        return LLMJudgeRewardFunction(**params)

    def create_task(self, prompt="Test prompt", model_output="Test output"):
        """Helper to create a task."""
        task = Task(
            task_id="test_task_1",
            task_name="Test Task",
            task_type="math",
            level=0,
            prompt=prompt,
            expected_answer="expected",
        )
        task.model_output = model_output
        return task

    @patch("transformers.AutoTokenizer.from_pretrained")
    def test_initialization(self, mock_tokenizer):
        """Test reward function initialization."""
        mock_tokenizer.return_value = MagicMock()

        rf = self.create_reward_function()
        self.assertFalse(rf.initialized)

        rf.initialize()
        self.assertTrue(rf.initialized)
        mock_tokenizer.assert_called_once_with(self.model_name)

    @patch("transformers.AutoTokenizer.from_pretrained")
    def test_initialization_failure(self, mock_tokenizer):
        """Test initialization failure handling."""
        mock_tokenizer.side_effect = RuntimeError("Model not found")

        rf = self.create_reward_function()
        with self.assertRaises(RuntimeError):
            rf.initialize()

    @patch("transformers.AutoTokenizer.from_pretrained")
    def test_format_conversation(self, mock_tokenizer):
        """Test conversation formatting."""
        mock_tokenizer.return_value = MagicMock()

        rf = self.create_reward_function()
        rf.initialize()

        prompt = "What is 2+2?"
        response = "The answer is 4."

        conv = rf._format_conversation(prompt, response)

        self.assertEqual(len(conv), 2)
        self.assertEqual(conv[0]["role"], "user")
        self.assertEqual(conv[0]["content"], prompt)
        self.assertEqual(conv[1]["role"], "assistant")
        self.assertEqual(conv[1]["content"], response)

    @patch("transformers.AutoTokenizer.from_pretrained")
    def test_apply_chat_template_without_bos(self, mock_tokenizer):
        """Test chat template application without BOS token."""
        mock_tok = MagicMock()
        mock_tok.apply_chat_template.return_value = "formatted text"
        mock_tok.bos_token = None
        mock_tokenizer.return_value = mock_tok

        rf = self.create_reward_function()
        rf.initialize()

        result = rf._apply_chat_template([{"role": "user", "content": "test"}])
        self.assertEqual(result, "formatted text")

    @patch("transformers.AutoTokenizer.from_pretrained")
    def test_apply_chat_template_with_bos(self, mock_tokenizer):
        """Test chat template application with BOS token removal."""
        mock_tok = MagicMock()
        mock_tok.apply_chat_template.return_value = "<s>formatted text"
        mock_tok.bos_token = "<s>"
        mock_tokenizer.return_value = mock_tok

        rf = self.create_reward_function()
        rf.initialize()

        result = rf._apply_chat_template([{"role": "user", "content": "test"}])
        self.assertEqual(result, "formatted text")

    @patch("transformers.AutoTokenizer.from_pretrained")
    @patch("requests.post")
    def test_call_judge_api_success(self, mock_post, mock_tokenizer):
        """Test successful API call."""
        mock_tokenizer.return_value = MagicMock()

        # Mock successful API response
        mock_response = MagicMock()
        mock_response.json.return_value = [
            {"embedding": [23.125]},
            {"embedding": [3.578125]},
        ]
        mock_post.return_value = mock_response

        rf = self.create_reward_function()
        rf.initialize()

        scores = rf._call_judge_api(["text1", "text2"])

        self.assertEqual(scores, [23.125, 3.578125])
        mock_post.assert_called_once()

    @patch("transformers.AutoTokenizer.from_pretrained")
    @patch("requests.post")
    def test_call_judge_api_connection_error(self, mock_post, mock_tokenizer):
        """Test API call with connection error."""
        mock_tokenizer.return_value = MagicMock()
        mock_post.side_effect = requests.exceptions.ConnectionError("Connection failed")

        rf = self.create_reward_function()
        rf.initialize()

        scores = rf._call_judge_api(["text1"])
        self.assertIsNone(scores)

    @patch("transformers.AutoTokenizer.from_pretrained")
    @patch("requests.post")
    def test_call_judge_api_malformed_response(self, mock_post, mock_tokenizer):
        """Test API call with malformed response."""
        mock_tokenizer.return_value = MagicMock()

        # Mock malformed response
        mock_response = MagicMock()
        mock_response.json.return_value = [{"no_embedding": []}]
        mock_post.return_value = mock_response

        rf = self.create_reward_function()
        rf.initialize()

        scores = rf._call_judge_api(["text1"])
        self.assertIsNone(scores)

    def test_normalize_score_disabled(self):
        """Test score normalization when disabled."""
        rf = self.create_reward_function(normalize=False)

        # Scores are clipped to [0, 1]
        self.assertEqual(rf._normalize_score(-10.0), 0.0)
        self.assertEqual(rf._normalize_score(0.5), 0.5)
        self.assertEqual(rf._normalize_score(100.0), 1.0)

    def test_normalize_score_enabled(self):
        """Test score normalization when enabled."""
        rf = self.create_reward_function(normalize=True)

        # Test specific score normalizations using tanh
        # Expected: (tanh(raw_score / 10) + 1) / 2
        normalized_0 = rf._normalize_score(0.0)
        self.assertAlmostEqual(normalized_0, 0.5, places=3)  # tanh(0) = 0

        normalized_pos = rf._normalize_score(10.0)
        self.assertGreater(normalized_pos, 0.5)  # tanh(1) ≈ 0.76

        normalized_neg = rf._normalize_score(-10.0)
        self.assertLess(normalized_neg, 0.5)  # tanh(-1) ≈ -0.76

        # All scores should be in [0, 1]
        self.assertGreaterEqual(normalized_pos, 0.0)
        self.assertLessEqual(normalized_pos, 1.0)
        self.assertGreaterEqual(normalized_neg, 0.0)
        self.assertLessEqual(normalized_neg, 1.0)

    @patch("transformers.AutoTokenizer.from_pretrained")
    def test_compute_reward_empty_prompt(self, mock_tokenizer):
        """Test reward computation with empty prompt."""
        mock_tokenizer.return_value = MagicMock()

        rf = self.create_reward_function()
        rf.initialize()

        task = self.create_task(prompt="", model_output="response")
        reward = rf.compute_reward(task)

        self.assertEqual(reward.score, 0.0)
        self.assertIn("prompt", reward.info.lower())

    @patch("transformers.AutoTokenizer.from_pretrained")
    def test_compute_reward_empty_output(self, mock_tokenizer):
        """Test reward computation with empty model output."""
        mock_tokenizer.return_value = MagicMock()

        rf = self.create_reward_function()
        rf.initialize()

        task = self.create_task(prompt="prompt", model_output="")
        reward = rf.compute_reward(task)

        self.assertEqual(reward.score, 0.0)
        self.assertIn("empty", reward.info.lower())

    @patch("transformers.AutoTokenizer.from_pretrained")
    @patch("requests.post")
    def test_compute_reward_successful(self, mock_post, mock_tokenizer):
        """Test successful reward computation."""
        mock_tok = MagicMock()
        mock_tok.apply_chat_template.return_value = "formatted"
        mock_tok.bos_token = None
        mock_tokenizer.return_value = mock_tok

        # Mock API response
        mock_response = MagicMock()
        mock_response.json.return_value = [{"embedding": [15.0]}]
        mock_post.return_value = mock_response

        rf = self.create_reward_function(normalize=True)
        rf.initialize()

        task = self.create_task(prompt="Q: 2+2?", model_output="A: 4")
        reward = rf.compute_reward(task)

        # compute_reward returns raw score (normalization done in batch method)
        self.assertEqual(reward.score, 15.0)
        self.assertIn("15.0", reward.info)

    @patch("transformers.AutoTokenizer.from_pretrained")
    @patch("requests.post")
    def test_compute_reward_below_threshold(self, mock_post, mock_tokenizer):
        """Test reward with score below threshold."""
        mock_tok = MagicMock()
        mock_tok.apply_chat_template.return_value = "formatted"
        mock_tok.bos_token = None
        mock_tokenizer.return_value = mock_tok

        # Mock API response with low score
        mock_response = MagicMock()
        mock_response.json.return_value = [{"embedding": [-150.0]}]
        mock_post.return_value = mock_response

        rf = self.create_reward_function(score_threshold=-100.0, normalize=True)
        rf.initialize()

        task = self.create_task(prompt="Q: test?", model_output="A: bad")
        reward = rf.compute_reward(task)

        self.assertEqual(reward.score, 0.0)
        self.assertIn("threshold", reward.info.lower())

    @patch("transformers.AutoTokenizer.from_pretrained")
    @patch("requests.post")
    def test_compute_reward_api_failure(self, mock_post, mock_tokenizer):
        """Test reward computation when API call fails."""
        mock_tok = MagicMock()
        mock_tok.apply_chat_template.return_value = "formatted"
        mock_tok.bos_token = None
        mock_tokenizer.return_value = mock_tok

        mock_post.side_effect = requests.exceptions.Timeout("API timeout")

        rf = self.create_reward_function()
        rf.initialize()

        task = self.create_task(prompt="Q: test?", model_output="A: response")
        reward = rf.compute_reward(task)

        self.assertEqual(reward.score, 0.0)
        self.assertIn("failed", reward.info.lower())

    @patch("transformers.AutoTokenizer.from_pretrained")
    def test_compute_reward_uninitialized(self, mock_tokenizer):
        """Test reward computation auto-initializes if needed."""
        mock_tok = MagicMock()
        mock_tok.apply_chat_template.return_value = "formatted"
        mock_tok.bos_token = None
        mock_tokenizer.return_value = mock_tok

        rf = self.create_reward_function()
        self.assertFalse(rf.initialized)

        # Call compute_reward without explicit initialize
        with patch("requests.post") as mock_post:
            mock_response = MagicMock()
            mock_response.json.return_value = [{"embedding": [1.0]}]
            mock_post.return_value = mock_response

            task = self.create_task()
            reward = rf.compute_reward(task)

            # Should be initialized now
            self.assertTrue(rf.initialized)

    @patch("transformers.AutoTokenizer.from_pretrained")
    @patch("requests.post")
    def test_realistic_scoring_scenario(self, mock_post, mock_tokenizer):
        """Test realistic scoring scenario matching the example in docstring."""
        mock_tok = MagicMock()
        mock_tok.apply_chat_template.return_value = "formatted"
        mock_tok.bos_token = None
        mock_tokenizer.return_value = mock_tok

        # Mock responses matching the example
        def api_side_effect(base_url, **kwargs):
            # Return correct score for response1, low score for response2
            texts = kwargs["json"]["text"]
            if len(texts) == 1:
                # Single call
                response = MagicMock()
                response.json.return_value = [{"embedding": [23.125]}]
                return response
            return None

        mock_post.side_effect = api_side_effect

        rf = self.create_reward_function(normalize=False)  # Keep raw scores
        rf.initialize()

        prompt = "Jane has 12 apples..."
        response1 = "1. Jane starts with 12 apples..."

        task = self.create_task(prompt=prompt, model_output=response1)
        reward = rf.compute_reward(task)

        # Raw score should be close to 23.125 (or normalized version)
        self.assertGreater(reward.score, 0.0)
        self.assertIn("23.125", reward.info)


class TestLLMJudgeIntegration(unittest.TestCase):
    """Integration tests for LLMJudgeRewardFunction."""

    @patch("transformers.AutoTokenizer.from_pretrained")
    @patch("requests.post")
    def test_multiple_tasks_scoring(self, mock_post, mock_tokenizer):
        """Test scoring multiple tasks in sequence."""
        mock_tok = MagicMock()
        mock_tok.apply_chat_template.return_value = "formatted"
        mock_tok.bos_token = None
        mock_tokenizer.return_value = mock_tok

        # Mock different scores for different calls
        scores_to_return = [[{"embedding": [10.0]}], [{"embedding": [5.0]}]]
        mock_post.side_effect = [MagicMock(json=lambda: s) for s in scores_to_return]

        rf = LLMJudgeRewardFunction(normalize=False)
        rf.initialize()

        task1 = Task(
            task_id="t1",
            task_name="Task 1",
            task_type="math",
            level=0,
            prompt="Q1",
            expected_answer="A1",
        )
        task1.model_output = "Response 1"

        task2 = Task(
            task_id="t2",
            task_name="Task 2",
            task_type="math",
            level=0,
            prompt="Q2",
            expected_answer="A2",
        )
        task2.model_output = "Response 2"

        reward1 = rf.compute_reward(task1)
        reward2 = rf.compute_reward(task2)

        # Both should have valid scores
        self.assertGreater(reward1.score, 0.0)
        self.assertGreater(reward2.score, 0.0)


if __name__ == "__main__":
    unittest.main()
