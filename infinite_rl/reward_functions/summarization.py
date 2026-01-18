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

        # Use the executor with qwen3_embed to compute similarity: (document=reference, query=candidate)
        try:
            stdout, stderr = self.executor.run_single(
                (reference, candidate), "qwen3_embed"
            )

            # stdout may be None on error (Executor returns None, "Executor Error: ...")
            if stdout is None:
                return RewardFunctionScore(
                    format_score=format_score,
                    correctness_score=0.0,
                    error_msg=stderr or "Executor Error",
                )

            # stdout can be JSON or raw numeric string; try to parse to float
            try:
                sim = float(str(stdout).strip())
                sim = max(0.0, min(1.0, sim))
            except Exception:
                # If not parseable, but JSON may be present; try to extract numbers
                import re

                m = re.search(r"([0-9]*\.?[0-9]+)", str(stdout))
                if m:
                    sim = float(m.group(1))
                    sim = max(0.0, min(1.0, sim))
                else:
                    return RewardFunctionScore(
                        format_score=format_score,
                        correctness_score=0.0,
                        error_msg=f"Could not parse similarity from qwen3 output: {stdout}",
                    )

            return RewardFunctionScore(format_score=format_score, correctness_score=sim)
        except Exception as e:
            return RewardFunctionScore(
                format_score=format_score,
                correctness_score=0.0,
                error_msg=f"Executor exception: {e}",
            )
