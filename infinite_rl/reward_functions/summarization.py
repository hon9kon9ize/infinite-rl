from typing import Union
from .reward_function import RewardFunction, RewardFunctionScore
from ..parser import ExampleParser
from ..executor import Executor


class SummarizationRewardFunction(RewardFunction):
    """Reward function that scores a model summary against a reference using
    a qwen3 embedding cosine similarity (document=reference, query=model_output).

    The function expects the model response to contain an <answer>...</answer> tag
    with the candidate summary.
    """

    def __init__(self, task_name: str = "summarization", timeout: int = 5):
        super().__init__(task_name, timeout=timeout)
        self.executor = None

    def initialize(self):
        self.executor = Executor(timeout=self.timeout)
        self.initialized = True

    def compute_reward(
        self,
        model_output: str,
        expected_output: Union[str, None],
        answer_tag: str = "answer",
    ) -> RewardFunctionScore:
        from ..parser import ExampleParser

        if not self.initialized:
            self.initialize()

        # 1. Format: require <answer> tags in model output
        matches = ExampleParser.extract_answer_tags(model_output, tags=answer_tag)
        if not matches:
            return RewardFunctionScore(
                format_score=0.0,
                correctness_score=0.0,
                error_msg=f"Missing <{answer_tag}> tags in response.",
            )

        candidate = matches[0].strip()
        format_score = 1.0

        # expected_output may already be a short language code or the reference text;
        # we treat it as the reference summary text to compare against. If it's empty, try to
        # extract from candidate (not ideal) and give full credit (conservative).
        reference = None
        if isinstance(expected_output, str) and expected_output.strip():
            reference = expected_output.strip()
        elif isinstance(expected_output, (int, float)):
            # Not meaningful
            reference = None

        if not reference:
            # Fallback: attempt to treat the whole model_output as a reference if it contains an <answer>
            # Already extracted candidate; without a proper reference we can't compute semantic similarity.
            return RewardFunctionScore(
                format_score=format_score,
                correctness_score=0.0,
                error_msg="No reference summary provided to compute similarity.",
            )

        # Use the executor to get embeddings for document and query separately
        try:
            # Document embedding
            doc_emb, doc_err = self.executor.get_embedding(reference, role="document")
            if doc_emb is None:
                return RewardFunctionScore(
                    format_score=format_score,
                    correctness_score=0.0,
                    error_msg=doc_err or "Executor Error",
                )

            # Query embedding
            qry_emb, qry_err = self.executor.get_embedding(candidate, role="query")
            if qry_emb is None:
                return RewardFunctionScore(
                    format_score=format_score,
                    correctness_score=0.0,
                    error_msg=qry_err or "Executor Error",
                )

            # Compute local cosine similarity
            try:
                sim = float(self.executor.cosine_similarity(doc_emb, qry_emb))
                sim = max(0.0, min(1.0, sim))
                return RewardFunctionScore(
                    format_score=format_score, correctness_score=sim
                )
            except Exception as e:
                return RewardFunctionScore(
                    format_score=format_score,
                    correctness_score=0.0,
                    error_msg=f"Cosine computation failed: {e}",
                )
        except Exception as e:
            return RewardFunctionScore(
                format_score=format_score,
                correctness_score=0.0,
                error_msg=f"Executor exception: {e}",
            )
