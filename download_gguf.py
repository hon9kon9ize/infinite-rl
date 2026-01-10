"""Utility script to download and setup Qwen3 embedding model GGUF file."""

import os
from pathlib import Path
from huggingface_hub import hf_hub_download


def download_qwen3_gguf():
    """Download jina-embeddings-v4-text-matching-Q4_K_M.gguf from Hugging Face."""

    # Create tmp directory if it doesn't exist
    tmp_dir = Path("tmp")
    tmp_dir.mkdir(exist_ok=True)

    model_id = "jinaai/jina-embeddings-v4-text-matching-GGUF"
    filename = "jina-embeddings-v4-text-matching-Q4_K_M.gguf"

    print(f"Downloading {filename} from {model_id}...")

    gguf_path = hf_hub_download(
        repo_id=model_id,
        filename=filename,
        local_dir=str(tmp_dir),
        cache_dir=str(tmp_dir),
    )

    print(f"✓ Downloaded successfully to: {gguf_path}")
    return gguf_path


if __name__ == "__main__":
    download_qwen3_gguf()
