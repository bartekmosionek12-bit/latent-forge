"""Success metric for EmbeddingRegionTarget: cosine distance to the region centroid."""
from __future__ import annotations

import torch


def cosine_distance(a: torch.Tensor, b: torch.Tensor) -> float:
    return 1.0 - torch.nn.functional.cosine_similarity(a.flatten(), b.flatten(), dim=0).item()


def evaluate_embedding_target(embedder, target, text: str) -> dict:
    c = target.centroid(embedder)
    e = embedder.encode([text], convert_to_tensor=True, normalize_embeddings=True)[0]
    return {"text": text, "cosine_distance_to_centroid": cosine_distance(e, c.to(e.device))}
