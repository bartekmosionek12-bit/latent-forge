"""External embedder (e.g. a RAG / memory retriever) loading, cached by name."""
from __future__ import annotations

from functools import lru_cache

import torch


@lru_cache(maxsize=4)
def load_embedder(name: str = "sentence-transformers/all-MiniLM-L6-v2", device: str = "auto"):
    from sentence_transformers import SentenceTransformer

    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    emb = SentenceTransformer(name, device=device)
    emb.eval()
    emb.requires_grad_(False)
    return emb
