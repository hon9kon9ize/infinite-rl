import os
import re
import json
import threading
from pathlib import Path
from typing import Union, Callable
from .reward_function import RewardFunction, RewardFunctionScore

# Defer heavy import to initialize() to speed up testing and avoid hangs during collection
SentenceTransformer = None


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
    def __init__(self, task_name: str = "summarization", timeout: int = 5):
        super().__init__(task_name, timeout=timeout)
        self._lock = threading.Lock()
        self.model = None

    def initialize(self):
        global SentenceTransformer
        if SentenceTransformer is None:
            from sentence_transformers import SentenceTransformer as ST

            SentenceTransformer = ST

        # Try to use GGUF model, fall back to default if not available
        gguf_file = _ensure_gguf_downloaded()

        model_kwargs = {}

        if gguf_file:
            model_kwargs["gguf_file"] = gguf_file

        try:
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
        from ..parser import ExampleParser

        # Handle expected_output being a JSON string (unwrap if necessary)
        if isinstance(expected_output, str):
            exp_matches = ExampleParser.extract_answer_tags(expected_output)
            if exp_matches:
                expected_output = exp_matches[0]

            trimmed = expected_output.strip()
            if trimmed.startswith("{") and trimmed.endswith("}"):
                try:
                    data = json.loads(trimmed)
                    for key in ["summary", "answer", "result", "text"]:
                        if key in data:
                            expected_output = data[key]
                            break
                except Exception:
                    pass

        # 1. Format Objective: Check for <answer> tags
        matches = ExampleParser.extract_answer_tags(model_output)

        if not matches:
            # Check if they used markdown code blocks instead
            if (
                "```summary" in model_output.lower()
                or "```text" in model_output.lower()
                or "```markdown" in model_output.lower()
            ):
                return RewardFunctionScore(
                    format_score=0.0,
                    correctness_score=0.0,
                    error_msg="Caught markdown code block (```summary) instead of required <answer> tags. Please use <answer>...</answer>.",
                )
            return RewardFunctionScore(
                format_score=0.0,
                correctness_score=0.0,
                error_msg="Missing <answer> tags in response. Ensure the final summary is wrapped in <answer> and </answer>.",
            )

        predicted_summary = matches[0] if matches else ""
        format_score = 1.0
        # If the tag was malformed but we extracted it, maybe give a slight penalty?
        # But for now, let's just accept it if it's there.
        if "<answer>" not in model_output.lower():
            format_score = 0.8  # Slight penalty for malformed tags

        error_msg = ""

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
                    error_msg = ""
                else:
                    correctness_score = pred_length / expected_output
                    error_msg = f"Summary too short ({pred_length} words). Minimum expected: {expected_output} words."
                return RewardFunctionScore(
                    format_score=format_score,
                    correctness_score=correctness_score,
                    error_msg=error_msg if correctness_score < 0.8 else "",
                )
            except Exception as e:
                return RewardFunctionScore(
                    format_score=format_score,
                    correctness_score=0.0,
                    error_msg=f"Error evaluating length: {e}",
                )

        else:
            # String: use semantic similarity
            try:
                with self._lock:
                    summary_embed = self.model.encode(predicted_summary.strip())
                    doc_embed = self.model.encode(expected_output.strip())
                    sim_result = self.model.similarity(summary_embed, doc_embed)

                    # Extract scalar value from similarity result (often a tensor or matrix)
                    if hasattr(sim_result, "item"):
                        similarity = sim_result.item()
                    else:
                        try:
                            similarity = float(sim_result[0][0])
                        except (IndexError, TypeError, KeyError):
                            similarity = float(sim_result)

                correctness_score = float(similarity)
            except Exception as e:
                print(f"Error computing embeddings: {e}")
                correctness_score = 0.0
                error_msg = f"Semantic similarity calculation failed: {e}"

            return RewardFunctionScore(
                format_score=format_score,
                correctness_score=correctness_score,
                error_msg=error_msg if correctness_score < 0.8 else "",
            )
