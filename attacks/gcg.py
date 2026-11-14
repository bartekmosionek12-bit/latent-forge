"""GCGAttack: thin wrapper over nanoGCG (GraySwanAI, v0.3.0).

API used (verified against the installed package source, nanogcg/gcg.py):
    nanogcg.run(model, tokenizer, messages: str | list[dict], target: str,
                config: GCGConfig | None) -> GCGResult(best_loss, best_string,
                                                        losses, strings)
``losses[i]`` is the lowest candidate loss sampled at step i; ``strings[i]`` is
the buffer-best suffix after step i. For base models without a chat template
nanogcg installs a plain-concatenation template on the tokenizer.
"""
from __future__ import annotations

import dataclasses
import time

import nanogcg
from nanogcg import GCGConfig

from targets import TextTarget

from .base import Attack, AttackResult

_GCG_FIELDS = {f.name for f in dataclasses.fields(GCGConfig)}


class GCGAttack(Attack):
    target_type = TextTarget

    def __init__(self, target: TextTarget, **config) -> None:
        unknown = set(config) - _GCG_FIELDS
        if unknown:
            raise TypeError(f"unknown GCGConfig fields: {sorted(unknown)}")
        super().__init__(target, **config)

    def run(self, bundle) -> AttackResult:
        cfg = GCGConfig(**self.config)
        t0 = time.time()
        res = nanogcg.run(bundle.model, bundle.tokenizer, self.target.prompt, self.target.text, cfg)
        return AttackResult(
            target=self.target,
            attack=type(self).__name__,
            payload=res.best_string,
            final_loss=res.best_loss,
            loss_history=list(res.losses),
            extra={"suffix_per_step": list(res.strings),
                   "seconds": time.time() - t0,
                   "gcg_config": {k: v for k, v in dataclasses.asdict(cfg).items()
                                  if k != "probe_sampling_config"}},
        )
