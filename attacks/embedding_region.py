from __future__ import annotations

from targets import EmbeddingRegionTarget

from .base import Attack, AttackResult


class EmbeddingRegionAttack(Attack):
    """Phase 3 (not implemented yet): gradient-guided token substitution towards
    a region of an external embedder (SentAttack-style), optional coherence term."""

    target_type = EmbeddingRegionTarget

    def run(self, bundle) -> AttackResult:
        raise NotImplementedError("EmbeddingRegionAttack lands in Phase 3")
