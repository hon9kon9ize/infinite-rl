SYSTEM_PROMPT = """
You are an expert data generator for Reinforcement Learning from Human Feedback (RLHF).
Your task is to generate high-quality synthetic data for training LLMs.
For each request, you must provide:
1. A prompt: A clear instruction or question.
2. A chosen response: A high-quality, accurate, and helpful response.
3. A rejected response: A response that is slightly worse, contains errors, or is less helpful than the chosen one.

Format your output as a JSON object with keys: "prompt", "chosen", "rejected".
"""

TYPE_PROMPTS = {
    "coding": "Generate a coding problem and two solutions (one correct/optimal, one with a bug or sub-optimal).",
    "math": "Generate a math problem and two solutions (one correct, one with a common calculation error).",
    "summarization": "Generate a short text and two summaries (one concise and accurate, one missing key details or slightly inaccurate).",
    "creativity": "Generate a creative writing prompt (e.g., story, poem) and two responses (one engaging and well-written, one bland or poorly structured).",
}
