from __future__ import annotations

from dataclasses import dataclass, field

import torch

from .base import Target


@dataclass(repr=False)
class EmbeddingRegionTarget(Target):
    """A region in an EXTERNAL embedder's space (e.g. a RAG / memory retriever),
    defined by example texts. The region centre is the normalised mean embedding."""

    reference_texts: list[str]
    embedder_name: str = "sentence-transformers/all-MiniLM-L6-v2"
    kind = "embedding_region"
    _centroid: torch.Tensor | None = field(default=None, init=False, compare=False)

    def __post_init__(self) -> None:
        if not self.reference_texts:
            raise ValueError("EmbeddingRegionTarget needs at least one reference text")

    def describe(self) -> str:
        return f"region of {len(self.reference_texts)} refs in {self.embedder_name}"

    def centroid(self, embedder) -> torch.Tensor:
        """Unit-norm centroid of the reference embeddings (cached).
        ``embedder`` is a sentence_transformers.SentenceTransformer."""
        if self._centroid is None:
            embs = embedder.encode(self.reference_texts, convert_to_tensor=True,
                                   normalize_embeddings=True)
            c = embs.mean(dim=0)
            self._centroid = c / c.norm()
        return self._centroid
