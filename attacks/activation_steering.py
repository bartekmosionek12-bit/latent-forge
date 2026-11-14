from __future__ import annotations

from targets import ActivationDirectionTarget

from .base import Attack, AttackResult


class ActivationSteeringAttack(Attack):
    """Phase 4 (not implemented yet): contrastive mean-difference direction in the
    residual stream, injected via forward hooks during generation."""

    target_type = ActivationDirectionTarget

    def run(self, bundle) -> AttackResult:
        raise NotImplementedError("ActivationSteeringAttack lands in Phase 4")
