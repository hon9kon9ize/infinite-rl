import os
import re
from pathlib import Path
from typing import Union, Callable
from sentence_transformers import SentenceTransformer
from .reward_function import RewardFunction, RewardFunctionScore


def _ensure_gguf_downloaded():
    """Ensure GGUF file is downloaded."""
    gguf_path = Path("tmp/jina-embeddings-v4-text-matching-Q4_K_M.gguf")

    if not gguf_path.exists():
        try:
            from huggingface_hub import hf_hub_download

            print("Downloading jina-embeddings-v4-text-matching-Q4_K_M.gguf...")
            gguf_path.parent.mkdir(exist_ok=True)

            hf_hub_download(
                repo_id="jinaai/jina-embeddings-v4-text-matching-GGUF",
                filename="jina-embeddings-v4-text-matching-Q4_K_M.gguf",
                local_dir=str(gguf_path.parent),
            )
            print(f"✓ Downloaded to {gguf_path}")
        except Exception as e:
            print(f"Warning: Could not download GGUF file: {e}")
            print("Falling back to default model...")
            return None

    return str(gguf_path)


class SummarizationRewardFunction(RewardFunction):
    model: SentenceTransformer

    def __init__(self, task_name: str = "summarization", timeout: int = 5):
        super().__init__(task_name, timeout=timeout)

    def initialize(self):
        # Try to use GGUF model, fall back to default if not available
        gguf_file = _ensure_gguf_downloaded()

        model_kwargs = {}

        if gguf_file:
            model_kwargs["gguf_file"] = gguf_file

        try:
            # self.model = SentenceTransformer(
            #     "Qwen/Qwen3-Embedding-0.6B",
            #     device="cpu",
            #     model_kwargs=model_kwargs,
            # )
            self.model = SentenceTransformer(
                "jinaai/jina-embeddings-v3", device="cpu", trust_remote_code=True
            )
        except Exception as e:
            print(f"Warning: Could not load Qwen model: {e}")
            print("Falling back to all-MiniLM-L6-v2...")
            self.model = SentenceTransformer("all-MiniLM-L6-v2", device="cpu")

        self.initialized = True

    def compute_reward(
        self,
        model_output: str,
        expected_output: Union[str, int, Callable],
        original_document: str = None,
    ) -> RewardFunctionScore:
        # 1. Format Objective: Check for <summary> tags
        tag_pattern = r"<summary>(.*?)</summary>"
        match = re.search(tag_pattern, model_output, re.DOTALL)

        if not match:
            return RewardFunctionScore(format_score=0.0, correctness_score=0.0)

        predicted_summary = match.group(1).strip()
        format_score = 1.0

        # Feature: Penalize length in format_score if original_document is provided
        # best score (1.0): < 50% of document length
        # worst score (0.0): >= 100% of document length (summary should be shorter than source)
        if original_document:
            source_len = len(original_document)
            summ_len = len(predicted_summary)
            if source_len > 0:
                ratio = summ_len / source_len
                if ratio < 0.5:
                    length_score = 1.0
                elif ratio >= 1.0:
                    length_score = 0.0
                else:
                    # Linear decay from 1.0 (at 0.5 ratio) to 0.0 (at 1.0 ratio)
                    # formula: 1.0 - (ratio - 0.5) / (1.0 - 0.5)
                    length_score = 1.0 - (ratio - 0.5) * 2.0

                # Combine tag presence (1.0) with length efficiency
                format_score = (format_score + length_score) / 2.0

        if not self.initialized:
            self.initialize()

        # Handle different expected_output types
        if callable(expected_output):
            # Callable: pass prediction to validator function
            try:
                result = expected_output(predicted_summary)
                if isinstance(result, bool):
                    correctness_score = 1.0 if result else 0.0
                elif isinstance(result, float):
                    correctness_score = result
                else:
                    correctness_score = 0.0
                return RewardFunctionScore(
                    format_score=format_score, correctness_score=correctness_score
                )
            except Exception as e:
                print(f"Error executing validator: {e}")
                return RewardFunctionScore(
                    format_score=format_score, correctness_score=0.0
                )

        elif isinstance(expected_output, int):
            # Int: not typical for summarization, but handle gracefully
            # Could represent minimum length or other metrics
            try:
                pred_length = len(predicted_summary.split())
                if pred_length >= expected_output:
                    correctness_score = 1.0
                else:
                    correctness_score = pred_length / expected_output
                return RewardFunctionScore(
                    format_score=format_score, correctness_score=correctness_score
                )
            except Exception:
                return RewardFunctionScore(
                    format_score=format_score, correctness_score=0.0
                )

        else:
            # String: use semantic similarity
            try:
                summary_embed = self.model.encode(predicted_summary.strip())
                doc_embed = self.model.encode(expected_output.strip())
                similarity = self.model.similarity(summary_embed, doc_embed).item()
                correctness_score = similarity
            except Exception as e:
                print(f"Error computing embeddings: {e}")
                correctness_score = 0.0

            return RewardFunctionScore(
                format_score=format_score, correctness_score=correctness_score
            )
