"""UniversalPageSuffixGCGAttack (topic injection, Phase 3): one suffix optimised jointly over
several product pages. It reuses PageSuffixGCGAttack for everything page-specific: one
sub-attack per page holds that page's KV-cached prefix, tail and target continuations, and
computes the page's loss terms exactly as in Phase 2. The joint objective is the MEAN over
pages of the per-page total (continuation CE + the arm's fluency / repetition / detector
terms); its gradient is the mean of the per-page gradients. A candidate is valid only if it
re-tokenises identically inside every page.

Targets per step: k in total, rotating over the pages (with k = number of pages, one target
per page per step, each page cycling through its own continuation set). The "full" metric
used to pick the best suffix is the mean over pages of each page's metric on its WHOLE
continuation set. The step loop, logging and resume are PageSuffixGCGAttack.run.
"""
from __future__ import annotations

import torch

from targets import UniversalPageInjectionTarget

from .base import Attack
from .page_suffix_gcg import DEFAULTS, PageSuffixGCGAttack


class UniversalPageSuffixGCGAttack(PageSuffixGCGAttack):
    target_type = UniversalPageInjectionTarget

    def __init__(self, target: UniversalPageInjectionTarget, **config) -> None:
        unknown = set(config) - set(DEFAULTS)
        if unknown:
            raise TypeError(f"unknown UniversalPageSuffixGCGAttack options: {sorted(unknown)}")
        if config.get("kl_weight", 0) > 0:
            raise ValueError("kl_weight is not defined for the universal attack")
        Attack.__init__(self, target, **{**DEFAULTS, **config})

    def setup(self, bundle) -> None:
        self.bundle, self.model, self.tok = bundle, bundle.model, bundle.tokenizer
        self.dev = bundle.device
        self.W = self.model.get_input_embeddings().weight
        self.subs = []
        for t in self.target.pages:
            sub = PageSuffixGCGAttack(t, **self.config)
            sub.setup(bundle)
            self.subs.append(sub)
        self.allowed = self.subs[0].allowed

    # ------------------------------------------------------------ targets per step
    def _n_targets(self) -> int:
        return sum(len(s.targets) for s in self.subs)

    def _step_targets(self, step: int, k: int, n_t: int):
        """k (page, target) pairs; consecutive slots go to consecutive pages."""
        npg = len(self.subs)
        out = []
        for i in range(k):
            slot = (step - 1) * k + i
            pg = slot % npg
            out.append([pg, (slot // npg) % len(self.subs[pg].targets)])
        return out

    def _by_page(self, spec):
        pages: dict[int, list[int]] = {}
        for pg, j in spec:
            pages.setdefault(int(pg), []).append(int(j))
        return pages

    # ------------------------------------------------------------ objective
    def score(self, cands: torch.Tensor, spec) -> dict:
        parts = [self.subs[pg].score(cands, idx) for pg, idx in self._by_page(spec).items()]
        return {k: torch.stack([p[k] for p in parts]).mean(0) for k in parts[0]}

    def token_grad(self, suffix: torch.Tensor, spec) -> torch.Tensor:
        by = self._by_page(spec)
        return sum(self.subs[pg].token_grad(suffix, idx) for pg, idx in by.items()) / len(by)

    def full_metric(self, suffix: torch.Tensor) -> dict:
        ms = [s.full_metric(suffix) for s in self.subs]
        out = {k: sum(m[k] for m in ms) / len(ms) for k in ms[0]}
        out["per_page"] = [m["total"] for m in ms]
        return out

    def roundtrip_ok(self, cands: torch.Tensor) -> torch.Tensor:
        ok = torch.ones(cands.shape[0], dtype=torch.bool)
        for s in self.subs:
            ok &= s.roundtrip_ok(cands)
        return ok

    def init_suffix(self) -> torch.Tensor:
        return self.subs[0].init_suffix()
