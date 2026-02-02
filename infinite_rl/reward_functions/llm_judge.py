"""LLM-as-a-Judge reward function using remote scoring model."""

import requests
from typing import Optional, TYPE_CHECKING
from .reward_function import RewardFunction, RewardFunctionScore

if TYPE_CHECKING:
    from ..task import Task


class LLMJudgeRewardFunction(RewardFunction):
    """Reward function that uses a remote LLM judge to score model responses.

    Uses the Skywork Reward Model (V2-Qwen3-4B) via sglang API endpoint to evaluate
    the quality of model responses. Scores are normalized to [0, 1] range.

    The score from the judge represents a continuous quality metric:
    - Higher scores indicate better responses
    - Scores are obtained from the model's classification endpoint
    - Scores are normalized and clipped to [0, 1] for consistency

    Args:
        task_name: Name of the task
        api_host: Host address of the sglang server (default: "localhost")
        api_port: Port of the sglang server (default: 8000)
        model_name: Model identifier for the judge (default: Skywork/Skywork-Reward-V2-Qwen3-4B)
        score_threshold: Threshold for parsing score validity (default: -100.0)
        normalize: Whether to normalize scores to [0, 1] (default: True)
        timeout: Timeout for reward function execution (default: 30)
        answer_tag: Tag used to extract answers from model responses
        think_tag: Tag used to extract reasoning from model responses
    """

    def __init__(
        self,
        task_name: str = "llm_judge",
        api_host: str = "localhost",
        api_port: int = 8000,
        model_name: str = "Skywork/Skywork-Reward-V2-Qwen3-4B",
        score_threshold: float = -100.0,
        normalize: bool = True,
        timeout: int = 30,
        answer_tag: str = "answer",
        think_tag: str = "think",
        target_tag: str = None,
    ):
        super().__init__(
            task_name,
            timeout=timeout,
            answer_tag=answer_tag,
            think_tag=think_tag,
            target_tag=target_tag,
        )
        self.api_host = api_host
        self.api_port = api_port
        self.model_name = model_name
        self.base_url = f"http://{api_host}:{api_port}/classify"
        self.score_threshold = score_threshold
        self.normalize = normalize
        self.tokenizer = None

    def initialize(self):
        """Initialize the tokenizer for the judge model."""
        if self.initialized:
            return

        try:
            from transformers import AutoTokenizer

            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.initialized = True
        except Exception as e:
            raise RuntimeError(
                f"Failed to initialize LLMJudgeRewardFunction: {e}. "
                f"Make sure transformers is installed and model '{self.model_name}' is available."
            )

    def _format_conversation(self, prompt: str, response: str) -> list:
        """Format prompt and response as a conversation."""
        return [
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": response},
        ]

    def _apply_chat_template(self, conversation: list) -> str:
        """Apply chat template to conversation."""
        if self.tokenizer is None:
            raise RuntimeError("Tokenizer not initialized. Call initialize() first.")

        formatted = self.tokenizer.apply_chat_template(conversation, tokenize=False)

        # Remove BOS token if present (as per original code)
        if self.tokenizer.bos_token is not None and formatted.startswith(
            self.tokenizer.bos_token
        ):
            formatted = formatted[len(self.tokenizer.bos_token) :]

        return formatted

    def _call_judge_api(self, formatted_texts: list) -> Optional[list]:
        """Call the judge API to get scores.

        Args:
            formatted_texts: List of formatted conversation strings

        Returns:
            List of scores or None if API call fails
        """
        try:
            payload = {
                "model": self.model_name,
                "text": formatted_texts,
            }

            response = requests.post(self.base_url, json=payload, timeout=self.timeout)
            response.raise_for_status()

            responses = response.json()
            scores = []

            for resp in responses:
                # Extract embedding[0] as the score
                if "embedding" in resp and len(resp["embedding"]) > 0:
                    scores.append(resp["embedding"][0])
                else:
                    return None

            return scores

        except requests.exceptions.RequestException as e:
            return None
        except (KeyError, IndexError, TypeError) as e:
            return None

    def _normalize_score(self, raw_score: float) -> float:
        """Normalize raw score to [0, 1] range.

        Uses tanh-based normalization: normalized = (tanh(raw_score / 10) + 1) / 2
        This maps extreme values to [0, 1] while preserving relative ordering.

        Args:
            raw_score: Raw score from judge model

        Returns:
            Normalized score in [0, 1] range
        """
        if not self.normalize:
            # If normalization disabled, clip to [0, 1]
            return max(0.0, min(1.0, raw_score))

        import math

        # Use tanh for smooth normalization
        normalized = (math.tanh(raw_score / 10.0) + 1.0) / 2.0
        return max(0.0, min(1.0, normalized))

    def compute_reward(
        self,
        task: "Task",
        **kwargs,
    ) -> RewardFunctionScore:
        """Compute reward using LLM judge.

        Args:
            task: Task object containing prompt, model_output, and metadata

        Returns:
            RewardFunctionScore with judge's evaluation
        """
        if not self.initialized:
            self.initialize()

        # Validate inputs
        if not task.prompt:
            return RewardFunctionScore(
                score=0.0,
                info="Task has no prompt.",
            )

        if not task.model_output:
            return RewardFunctionScore(
                score=0.0,
                info="Model output is empty.",
            )

        try:
            # Format conversation
            conversation = self._format_conversation(task.prompt, task.model_output)

            # Apply chat template
            formatted_text = self._apply_chat_template(conversation)

            # Call judge API
            scores = self._call_judge_api([formatted_text])

            if scores is None or len(scores) == 0:
                return RewardFunctionScore(
                    score=0.0,
                    info="Failed to get score from judge API.",
                )

            raw_score = scores[0]

            # Check if score is below threshold
            if raw_score < self.score_threshold:
                return RewardFunctionScore(
                    score=0.0,
                    info=f"Judge score {raw_score:.4f} is below threshold {self.score_threshold}.",
                )

            # Normalize and return score
            normalized_score = self._normalize_score(raw_score)

            return RewardFunctionScore(
                score=normalized_score,
                info=f"Judge score: {raw_score:.4f} → normalized: {normalized_score:.4f}",
            )

        except Exception as e:
            return RewardFunctionScore(
                score=0.0,
                info=f"Error computing judge reward: {str(e)}",
            )
