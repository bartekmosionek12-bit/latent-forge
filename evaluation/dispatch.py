"""Pick the right success metric for an AttackResult based on its target type."""
from __future__ import annotations

from targets import ActivationDirectionTarget, EmbeddingRegionTarget, TextTarget

from .embedding import evaluate_embedding_target
from .text import evaluate_text_target


def evaluate_result(result, bundle) -> dict:
    t = result.target
    if isinstance(t, TextTarget):
        return evaluate_text_target(bundle, t, result.payload)
    if isinstance(t, EmbeddingRegionTarget):
        from models import load_embedder

        emb = load_embedder(t.embedder_name)
        report = {}
        payload = result.extra.get("payload_text", "")
        init_span = result.extra.get("init_span_text")
        if payload:
            report["payload_only"] = evaluate_embedding_target(emb, t, payload)
        if init_span is not None:
            report["payload_plus_init_span"] = evaluate_embedding_target(
                emb, t, f"{payload} {init_span}".strip())
        report["after_attack"] = evaluate_embedding_target(emb, t, result.payload)
        return report
    if isinstance(t, ActivationDirectionTarget):
        from .activation import activation_projection

        d, L = result.payload, t.layer
        report = {
            "reference": {
                "positive_prompts_mean_proj": _mean(activation_projection(bundle, p, d, L)
                                                    for p in t.positive_prompts),
                "negative_prompts_mean_proj": _mean(activation_projection(bundle, p, d, L)
                                                    for p in t.negative_prompts),
            },
            "by_alpha": {},
        }
        # generated continuations are re-read by the UNSTEERED model and projected
        for alpha, gens in result.extra["generations"].items():
            projs = [activation_projection(bundle, g, d, L) for g in gens.values() if g.strip()]
            report["by_alpha"][alpha] = {"mean_projection_of_output": _mean(projs),
                                         "generations": gens}
        return report
    raise TypeError(f"no evaluator for {type(t).__name__}")


def _mean(xs) -> float:
    xs = list(xs)
    return sum(xs) / len(xs) if xs else float("nan")
