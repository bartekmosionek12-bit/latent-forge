"""Pick the right success metric for an AttackResult based on its target type."""
from __future__ import annotations

from targets import ActivationDirectionTarget, EmbeddingRegionTarget, TextTarget

from .text import evaluate_text_target


def evaluate_result(result, bundle) -> dict:
    t = result.target
    if isinstance(t, TextTarget):
        return evaluate_text_target(bundle, t, result.payload)
    if isinstance(t, EmbeddingRegionTarget):
        raise NotImplementedError("embedding-region evaluation wiring lands in Phase 3")
    if isinstance(t, ActivationDirectionTarget):
        raise NotImplementedError("activation-direction evaluation wiring lands in Phase 4")
    raise TypeError(f"no evaluator for {type(t).__name__}")
