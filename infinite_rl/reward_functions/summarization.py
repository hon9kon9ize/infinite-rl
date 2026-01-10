import re
import torch
from sentence_transformers import SentenceTransformer, util
from .reward_function import RewardFunction


class SummarizationRewardFunction(RewardFunction):
    model: SentenceTransformer

    def initialize(self):
        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = SentenceTransformer("Qwen/Qwen3-Embedding-0.6B", device=device)
        self.initialized = True

    def compute_reward(self, model_output: str, reference_answer) -> float:
        # 1. Format Objective: Check for <answer> tags
        tag_pattern = r"<answer>(.*?)</answer>"
        match = re.search(tag_pattern, model_output, re.DOTALL)

        if not match:
            return (0.0, 0.0)

        format_score = 1.0
        predicted_summary = match.group(1).strip()

        if not self.initialized:
            raise RuntimeError("Reward function not initialized.")

        # 2. Correctness (Semantic Similarity) Objective
        try:
            emb_pred = self.model.encode(predicted_summary)
            emb_ref = self.model.encode(reference_answer)
            correctness_score = util.cos_sim(emb_pred, emb_ref).item()
        except Exception:
            correctness_score = 0.0

        return (format_score, correctness_score)
