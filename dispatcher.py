"""Goal -> strategy layer: given a Target, pick and configure the Attack.

This is the piece existing frameworks (AdversariaLLM, LLMart) don't have: the
user states WHERE in latent space they want to land, the dispatcher decides HOW.
Minimal version: type-based registry with isinstance matching (MRO-aware, so a
subclass of TextTarget still routes to GCGAttack unless registered itself).
"""
from __future__ import annotations

from typing import Any

from attacks import ActivationSteeringAttack, Attack, AttackResult, EmbeddingRegionAttack, GCGAttack
from targets import ActivationDirectionTarget, EmbeddingRegionTarget, Target, TextTarget

ATTACK_REGISTRY: dict[type[Target], type[Attack]] = {
    TextTarget: GCGAttack,
    EmbeddingRegionTarget: EmbeddingRegionAttack,
    ActivationDirectionTarget: ActivationSteeringAttack,
}


def register(target_type: type[Target], attack_type: type[Attack]) -> None:
    """Plug in a new Target/Attack pair."""
    ATTACK_REGISTRY[target_type] = attack_type


def select_attack_class(target: Target) -> type[Attack]:
    for cls in type(target).__mro__:
        if cls in ATTACK_REGISTRY:
            return ATTACK_REGISTRY[cls]
    raise TypeError(f"no attack registered for target type {type(target).__name__}")


def select_attack(target: Target, **attack_config: Any) -> Attack:
    """Return a configured Attack instance for ``target``."""
    return select_attack_class(target)(target, **attack_config)


def run(target: Target, bundle, evaluate: bool = True, **attack_config: Any):
    """Full cycle: dispatch -> attack -> (optional) evaluation.
    Returns (AttackResult, evaluation dict | None)."""
    attack = select_attack(target, **attack_config)
    result: AttackResult = attack.run(bundle)
    report = None
    if evaluate:
        from evaluation import evaluate_result
        report = evaluate_result(result, bundle)
    return result, report
