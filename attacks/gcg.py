"""Phase 2 placeholder; real implementation added in the next commit."""
from __future__ import annotations

from targets import TextTarget

from .base import Attack, AttackResult


class GCGAttack(Attack):
    target_type = TextTarget

    def run(self, bundle) -> AttackResult:
        raise NotImplementedError("GCGAttack lands in Phase 2")
