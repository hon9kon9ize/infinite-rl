"""LLM-as-a-Judge reward function using remote scoring model."""

from copy import deepcopy
import requests
import time
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
        self, prompt, response: str, system_prompt: Optional[str] = None
    ) -> list:
        """Format prompt and response as a conversation."""
        conversation = []

        if system_prompt:
            conversation.append({"role": "system", "content": system_prompt})

        if isinstance(prompt, list):
            conversation.extend(deepcopy(prompt))
            conversation.append({"role": "assistant", "content": response})
        else:
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

        FIX #9: Added retry logic with exponential backoff and timeout
        to handle transient network issues.

        Args:
            formatted_texts: List of formatted conversation strings

        Returns:
            List of scores or None if API call fails after retries
        """
        # FIX #9: Retry logic with exponential backoff
        max_retries = 3
        base_delay = 2  # seconds

        for attempt in range(max_retries):
            try:
                payload = {
                    "model": self.model_name,
                    "text": formatted_texts,
                }
                # Use a longer timeout for batch API calls (60 seconds)
                response = requests.post(
                    self.base_url, json=payload, timeout=60
                )
                response.raise_for_status()
                responses = response.json()
                scores = []

                for resp in responses:
                    scores.append(resp["embedding"][0])

                assert len(scores) == len(
                    formatted_texts
                ), "Mismatch in number of scores returned"

                return scores

            except (requests.exceptions.Timeout, requests.exceptions.ConnectionError) as e:
                if attempt < max_retries - 1:
                    # Exponential backoff: wait longer on each retry
                    delay = base_delay * (2 ** attempt)
                    print(
                        f"Warning: LLM Judge API call failed (attempt {attempt + 1}/{max_retries}), "
                        f"retrying in {delay}s... Error: {e}"
                    )
                    time.sleep(delay)
                else:
                    # Last attempt failed
                    return None

            except (KeyError, IndexError, TypeError) as e:
                # Parse errors - don't retry
                return None

            except requests.exceptions.RequestException as e:
                # Other network errors - try again
                if attempt < max_retries - 1:
                    delay = base_delay * (2 ** attempt)
                    print(
                        f"Warning: LLM Judge API call failed (attempt {attempt + 1}/{max_retries}), "
                        f"retrying in {delay}s... Error: {e}"
                    )
                    time.sleep(delay)
                else:
                    return None

        return None

    def _normalize_score(self, raw_score: float) -> float:
        """Normalize raw score to [0, 1] range.

        Uses simple sigmoid-like normalization: score / (score + k)
        where k = 20/9 ensures that score 20 maps to 0.9

        Args:
            raw_score: Raw score from judge model

        Returns:
            Normalized score in [0, 1] range
        """
        if not self.normalize:
            # If normalization disabled, clip to [0, 1]
            return max(0.0, min(1.0, raw_score))

        # k = 20/9 ensures that 20 maps to 0.9
        k = 20 / 9

        if raw_score < 0:
            return 0.0

        return raw_score / (raw_score + k)

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

        # Apply correctness gating for math and puzzle tasks
        is_correct = kwargs.get("is_correct", False)
        if task.task_type in ["math", "puzzle"] and not is_correct:
            return RewardFunctionScore(
                score=0.0,
                info="LLM Judge reward gated: generation is incorrect.",
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

            # Return raw score (normalization done in batch method)
            return RewardFunctionScore(
                score=raw_score,
                info=f"Judge score: {raw_score:.4f} (raw)",
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
            List of lists of RewardFunctionScore objects, one list per task with scores for each generation
        """
        if not self.initialized:
            self.initialize()

        if not tasks:
            return []

        try:
            # Prepare all formatted texts for batch API call
            formatted_texts = []
            task_gen_data = []  # Store (task_idx, gen_idx) for each text

            for task_idx, task in enumerate(tasks):
                for gen_idx, gen in enumerate(task.generations):
                    # Validate generation has output
                    if not gen.output:
                        continue

                    # Format conversation for this generation
                    judge_system_prompt = getattr(task, "judge_system_prompt", None)
                    conversation = self._format_conversation(
                        task.prompt, gen.output, judge_system_prompt
                    )

                    # Apply chat template
                    formatted_text = self._apply_chat_template(conversation)
                    formatted_texts.append(formatted_text)
                    task_gen_data.append((task_idx, gen_idx))

            # Call batch judge API
            scores = self._call_judge_api(formatted_texts)

            if scores is None or len(scores) != len(formatted_texts):
                # API call failed - return empty lists for all tasks
                return [[] for _ in tasks]

            # Group scores by task
            task_scores = [[] for _ in tasks]
            for (task_idx, gen_idx), score in zip(task_gen_data, scores):
                task = tasks[task_idx]
                gen = task.generations[gen_idx]

                # Apply correctness gating for math and puzzle tasks at generation level
                if task.task_type in ["math", "puzzle"] and not gen.is_correct:
                    reward_score = RewardFunctionScore(
                        score=0.0,
                        info="LLM Judge reward gated: generation is incorrect.",
                    )
                else:
                    # Check if score is below threshold
                    if score < self.score_threshold:
                        reward_score = RewardFunctionScore(
                            score=0.0,
                            info=f"Judge score {score:.4f} is below threshold {self.score_threshold}.",
                        )
                    else:
                        # Normalize and create result
                        normalized_score = self._normalize_score(score)
                        reward_score = RewardFunctionScore(
                            score=normalized_score,
                            info=f"Judge score: {score:.4f} → normalized: {normalized_score:.4f}",
                        )
                task_scores[task_idx].append((gen_idx, reward_score))

            # Sort by generation index and return just the scores
            result = []
            for task_scores_list in task_scores:
                sorted_scores = [score for _, score in sorted(task_scores_list)]
                result.append(sorted_scores)

            return result

        except Exception as e:
            # On error, return empty lists
            return [[] for _ in tasks]
