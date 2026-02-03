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
        """Initialize the tokenizer for the judge model.

        Raises ImportError if transformers is not available. Other exceptions
        from tokenizer loading are propagated to allow proper error handling.

        Can be called multiple times - will return early if already initialized.

        Args:
            None

        Returns:
            None

        Raises:
            ImportError: If transformers is not installed
            Other exceptions from AutoTokenizer.from_pretrained propagate
        """
        if self.initialized:
            return

        try:
            from transformers import AutoTokenizer
        except ImportError:
            # transformers not installed - issue warning and continue
            # Tests can manually set tokenizer with a mock
            import warnings

            warnings.warn(
                f"transformers library not available. "
                f"LLMJudgeRewardFunction.tokenizer will remain None. "
                f"Tests can mock the tokenizer with: rf.tokenizer = MockTokenizer(). "
                f"Production requires: pip install transformers"
            )
            self.initialized = True  # Mark as initialized to avoid repeated attempts
            return

        # If we get here, transformers is available
        # Now load the specific tokenizer - let exceptions from this propagate
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.initialized = True

    def _format_conversation(
        self, prompt: str, response: str, system_prompt: Optional[str] = None
    ) -> list:
        """Format prompt and response as a conversation."""
        conversation = []

        if system_prompt:
            conversation.append({"role": "system", "content": system_prompt})

        conversation.extend(
            [
                {"role": "user", "content": prompt},
                {"role": "assistant", "content": response},
            ]
        )

        return conversation

    def _apply_chat_template(self, conversation: list) -> str:
        """Apply chat template to conversation.

        Requires tokenizer to be initialized (either real or mocked in tests).
        """
        if self.tokenizer is None:
            raise RuntimeError(
                "Tokenizer not initialized. Either transformers is not installed "
                "(install with: pip install transformers), or you need to mock "
                "the tokenizer in your tests."
            )

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
            # Use judge_system_prompt if available (for truthy tasks), otherwise None
            judge_system_prompt = getattr(task, "judge_system_prompt", None)
            conversation = self._format_conversation(
                task.prompt, task.model_output, judge_system_prompt
            )

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

    def compute_rewards_batch(
        self,
        tasks: list,
        **kwargs,
    ) -> list:
        """Compute rewards for multiple tasks in batch using LLM judge.

        This method enables efficient batched API calls by processing multiple tasks
        in a single request to the judge API.

        Args:
            tasks: List of Task objects to evaluate

        Returns:
            List of RewardFunctionScore objects in the same order as input tasks
        """
        if not self.initialized:
            self.initialize()

        if not tasks:
            return []

        try:
            # Prepare all formatted texts for batch API call
            formatted_texts = []
            task_data = []  # Store task data for error handling

            for task in tasks:
                # Validate inputs
                if not task.prompt or not task.model_output:
                    # Invalid task - mark for later
                    formatted_texts.append(None)
                    task_data.append((task, None))
                    continue

                # Format conversation
                judge_system_prompt = getattr(task, "judge_system_prompt", None)
                conversation = self._format_conversation(
                    task.prompt, task.model_output, judge_system_prompt
                )

                # Apply chat template
                formatted_text = self._apply_chat_template(conversation)
                formatted_texts.append(formatted_text)
                task_data.append((task, formatted_text))

            # Call batch judge API (filter out None values)
            valid_indices = [
                i for i, text in enumerate(formatted_texts) if text is not None
            ]
            valid_texts = [formatted_texts[i] for i in valid_indices]

            if not valid_texts:
                # All tasks were invalid
                return [
                    RewardFunctionScore(
                        score=0.0,
                        info="Task has no prompt or model output.",
                    )
                    for _ in tasks
                ]

            scores = self._call_judge_api(valid_texts)

            if scores is None or len(scores) != len(valid_texts):
                # API call failed - return default scores for all tasks
                return [
                    RewardFunctionScore(
                        score=0.0,
                        info="Failed to get scores from judge API.",
                    )
                    for _ in tasks
                ]

            # Build results array in original task order
            results = []
            valid_score_idx = 0

            for i, (task, formatted_text) in enumerate(task_data):
                if formatted_text is None:
                    # This task was invalid
                    results.append(
                        RewardFunctionScore(
                            score=0.0,
                            info="Task has no prompt or model output.",
                        )
                    )
                else:
                    raw_score = scores[valid_score_idx]
                    valid_score_idx += 1

                    # Check if score is below threshold
                    if raw_score < self.score_threshold:
                        results.append(
                            RewardFunctionScore(
                                score=0.0,
                                info=f"Judge score {raw_score:.4f} is below threshold {self.score_threshold}.",
                            )
                        )
                    else:
                        # Normalize and add result
                        normalized_score = self._normalize_score(raw_score)
                        results.append(
                            RewardFunctionScore(
                                score=normalized_score,
                                info=f"Judge score: {raw_score:.4f} → normalized: {normalized_score:.4f}",
                            )
                        )

            return results

        except Exception as e:
            # Return error score for all tasks
            return [
                RewardFunctionScore(
                    score=0.0,
                    info=f"Error computing batch judge rewards: {str(e)}",
                )
                for _ in tasks
            ]
