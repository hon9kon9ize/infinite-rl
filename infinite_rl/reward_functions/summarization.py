import os
import re
from pathlib import Path
from typing import Union, Callable
from sentence_transformers import SentenceTransformer
from .reward_function import RewardFunction


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
        self, model_output: str, reference_answer: Union[str, int, Callable]
    ) -> float:
        # 1. Format Objective: Check for <summary> tags
        tag_pattern = r"<summary>(.*?)</summary>"
        match = re.search(tag_pattern, model_output, re.DOTALL)

        if not match:
            return (0.0, 0.0)

        format_score = 1.0
        predicted_summary = match.group(1).strip()

        if not self.initialized:
            raise RuntimeError("Reward function not initialized.")

        # Handle different reference_answer types
        if callable(reference_answer):
            # Callable: pass prediction to validator function
            try:
                result = reference_answer(predicted_summary)
                if isinstance(result, bool):
                    correctness_score = 1.0 if result else 0.0
                elif isinstance(result, float):
                    correctness_score = result
                else:
                    correctness_score = 0.0
                return (format_score, correctness_score)
            except Exception as e:
                print(f"Error executing validator: {e}")
                return (format_score, 0.0)

        elif isinstance(reference_answer, int):
            # Int: not typical for summarization, but handle gracefully
            # Could represent minimum length or other metrics
            try:
                pred_length = len(predicted_summary.split())
                if pred_length >= reference_answer:
                    correctness_score = 1.0
                else:
                    correctness_score = pred_length / reference_answer
                return (format_score, correctness_score)
            except Exception:
                return (format_score, 0.0)

        else:
            # String: use semantic similarity
            try:
                print("prompt")
                # summary_embed = self.model.encode(
                #     predicted_summary.strip(), prompt="summarization"
                # )
                summary_embed = self.model.encode(predicted_summary.strip())
                doc_embed = self.model.encode(reference_answer.strip())
                correctness_score = self.model.similarity(
                    summary_embed, doc_embed
                ).item()
            except Exception as e:
                print(f"Error computing embeddings: {e}")
                correctness_score = 0.0

            return (format_score, correctness_score)
