"""GCGRegionAttack: GCG-style discrete suffix search whose objective is a REGION in
the attacked model's hidden-state space, not a fixed output string.

Why not nanoGCG: nanogcg 0.3.0 hard-codes cross-entropy on target tokens in
compute_token_gradient / _compute_candidates_loss_* (see docs/region_gcg_design.md).

Input layout  : [before] [suffix (optimised, n tokens)] [after]
                (split_prompt: before = prompt, after = "" for base models; the
                chat-template user/assistant wrapper for chat models)
Objective     : 1 - cos(h_L(last token), centroid) where h_L is the output of
                decoder block L (same hook as ActivationSteeringAttack).
Search        : grad w.r.t. one-hot(suffix) -> top-k swaps per position ->
                ``search_width`` single-swap candidates -> exact batched scoring ->
                accept the best candidate if it improves AND the full string
                (before + suffix + after) re-tokenises to exactly the same ids.
"""
from __future__ import annotations

import time

import torch

from models import get_decoder_layers, split_prompt
from targets import HiddenStateRegionTarget

from .base import Attack, AttackResult

DEFAULTS = dict(
    n_tokens=20,
    init="x",                  # filler word repeated n_tokens times
    num_steps=200,
    topk=128,
    search_width=256,
    seed=0,
    # Phase 2A: "min" = minimise distance (Phase 1); "band" = minimise |d - radius|,
    # i.e. land at the distance where real topical texts sit instead of overshooting.
    objective="min",
    target_radius="auto",      # "auto" = midpoint of the reference texts' own range
    # Phase 2B: words whose subword tokens may NOT be used in the suffix
    blocked_words=(),
)


class GCGRegionAttack(Attack):
    target_type = HiddenStateRegionTarget

    def __init__(self, target: HiddenStateRegionTarget, **config) -> None:
        unknown = set(config) - set(DEFAULTS)
        if unknown:
            raise TypeError(f"unknown GCGRegionAttack options: {sorted(unknown)}")
        super().__init__(target, **{**DEFAULTS, **config})

    # ---- setup --------------------------------------------------------------
    def setup(self, bundle) -> None:
        self.bundle = bundle
        self.tok = bundle.tokenizer
        self.dev = bundle.device
        self.W = bundle.model.get_input_embeddings().weight
        self.block = get_decoder_layers(bundle.model)[self.target.layer]
        parts = split_prompt(self.tok, self.target.prompt)
        self.before_str, self.after_str = parts.before, parts.after
        self.before_ids = self._ids(self.before_str, special=True)
        self.after_ids = self._ids(self.after_str)
        self.centroid = self.target.centroid(bundle).to(self.dev)
        self.allowed = allowed_token_mask(self.tok, self.W.shape[0]).to(self.dev)
        self.blocked_ids = blocked_token_ids(self.tok, self.config["blocked_words"])
        if self.blocked_ids:
            self.allowed[torch.tensor(sorted(self.blocked_ids), device=self.dev)] = False
        self.reference_distances = None
        self.radius = None
        if self.config["objective"] == "band":
            from evaluation.hidden_region import last_token_distance

            self.reference_distances = [last_token_distance(bundle, r, self.target.layer, self.centroid)
                                        for r in self.target.reference_texts]
            r = self.config["target_radius"]
            self.radius = ((min(self.reference_distances) + max(self.reference_distances)) / 2
                           if r == "auto" else float(r))
        elif self.config["objective"] != "min":
            raise ValueError(f"unknown objective {self.config['objective']!r}")

    def _ids(self, text: str, special: bool = False) -> torch.Tensor:
        if not text:
            return torch.empty(0, dtype=torch.long, device=self.dev)
        return self.tok(text, add_special_tokens=special, return_tensors="pt")["input_ids"][0].to(self.dev)

    def full_ids(self, suffix: torch.Tensor) -> torch.Tensor:
        return torch.cat([self.before_ids, suffix, self.after_ids])

    def full_text(self, suffix: torch.Tensor) -> str:
        return self.before_str + self.tok.decode(suffix) + self.after_str

    # ---- objective ----------------------------------------------------------
    def _last_state(self, **model_inputs) -> torch.Tensor:
        cap = {}

        def hook(_m, _a, out):
            cap["h"] = out[0] if isinstance(out, tuple) else out

        h = self.block.register_forward_hook(hook)
        try:
            self.bundle.model(**model_inputs)
        finally:
            h.remove()
        return cap["h"][:, -1, :].float()                         # [B, d]

    def distance_ids(self, ids: torch.Tensor) -> torch.Tensor:
        """Batched exact cosine distance for [B, L] ids."""
        with torch.no_grad():
            h = self._last_state(input_ids=ids)
        return 1 - torch.nn.functional.cosine_similarity(h, self.centroid[None], dim=-1)

    def objective(self, d: torch.Tensor) -> torch.Tensor:
        return d if self.radius is None else (d - self.radius).abs()

    def loss_ids(self, ids: torch.Tensor) -> torch.Tensor:
        """Batched exact objective for [B, L] ids."""
        return self.objective(self.distance_ids(ids))

    def loss_from_suffix_embeds(self, suffix_embeds: torch.Tensor) -> torch.Tensor:
        """Differentiable loss from [n, d] suffix embeddings (one sequence)."""
        emb = torch.cat([self.W[self.before_ids], suffix_embeds, self.W[self.after_ids]])
        h = self._last_state(inputs_embeds=emb[None])[0]
        return self.objective(1 - torch.nn.functional.cosine_similarity(h, self.centroid, dim=0))

    def token_grad(self, suffix: torch.Tensor) -> torch.Tensor:
        onehot = torch.nn.functional.one_hot(suffix, self.W.shape[0]).to(self.W.dtype)
        onehot.requires_grad_(True)
        loss = self.loss_from_suffix_embeds(onehot @ self.W)
        return torch.autograd.grad(loss, onehot)[0]                # [n, V]

    def roundtrip_ok(self, suffix: torch.Tensor) -> bool:
        again = self._ids(self.full_text(suffix), special=True)
        full = self.full_ids(suffix)
        return again.numel() == full.numel() and bool((again == full).all())

    def init_suffix(self) -> torch.Tensor:
        filler = self._ids(" " + self.config["init"])
        return filler.repeat(self.config["n_tokens"])[: self.config["n_tokens"]]

    # ---- main loop ----------------------------------------------------------
    def run(self, bundle) -> AttackResult:
        c, t0 = self.config, time.time()
        torch.manual_seed(c["seed"])
        self.setup(bundle)
        suffix = self.init_suffix()
        init_suffix = suffix.clone()
        cur = self.loss_ids(self.full_ids(suffix)[None]).item()
        history, accepted = [cur], 0
        suffix_history = [self.tok.decode(suffix)]   # suffix after each step (index = step)
        dist_history = [self.distance_ids(self.full_ids(suffix)[None]).item()]

        for _ in range(c["num_steps"]):
            grad = self.token_grad(suffix)
            grad[:, ~self.allowed] = float("inf")
            top = (-grad).topk(c["topk"], dim=1).indices
            pos = torch.randint(0, suffix.numel(), (c["search_width"],), device=self.dev)
            new = top[pos, torch.randint(0, c["topk"], (c["search_width"],), device=self.dev)]
            cands = suffix.repeat(c["search_width"], 1)
            cands[torch.arange(c["search_width"]), pos] = new
            cands = torch.unique(cands, dim=0)
            cands = cands[(cands != suffix).any(dim=1)]
            losses = self.loss_ids(torch.stack([self.full_ids(s) for s in cands]))
            for i in losses.argsort().tolist():
                if losses[i].item() >= cur:
                    break
                if self.roundtrip_ok(cands[i]):
                    suffix, cur, accepted = cands[i], losses[i].item(), accepted + 1
                    break
            history.append(cur)
            suffix_history.append(self.tok.decode(suffix))
            dist_history.append(self.distance_ids(self.full_ids(suffix)[None]).item())

        suffix_text = self.tok.decode(suffix)
        return AttackResult(
            target=self.target, attack=type(self).__name__, payload=suffix_text,
            final_loss=dist_history[-1], loss_history=history,
            extra={"init_suffix": self.tok.decode(init_suffix), "accepted_swaps": accepted,
                   "suffix_history": suffix_history, "distance_history": dist_history,
                   "objective": c["objective"], "radius": self.radius,
                   "reference_distances": self.reference_distances,
                   "n_blocked_tokens": len(self.blocked_ids),
                   "blocked_tokens_in_suffix": [self.tok.decode([i]) for i in suffix.tolist()
                                                if i in self.blocked_ids],
                   "seconds": time.time() - t0, "config": dict(c)},
        )


def allowed_token_mask(tokenizer, vocab_size: int) -> torch.Tensor:
    """ASCII tokens with at least one letter/digit, no special tokens."""
    mask = torch.zeros(vocab_size, dtype=torch.bool)
    special = set(tokenizer.all_special_ids)
    for i in range(min(vocab_size, len(tokenizer))):
        if i in special:
            continue
        s = tokenizer.decode([i])
        if s.isascii() and any(ch.isalnum() for ch in s) and "\n" not in s:
            mask[i] = True
    return mask


def blocked_token_ids(tokenizer, words) -> set[int]:
    """All vocab tokens that are part of, or contain, the given topic words.

    1. every token produced when tokenising each word in common surface forms
       (lower / Capitalised / UPPER, with and without a leading space);
    2. every token whose letters (lower-cased, >= 4 chars) are a substring of a word,
       which catches subword pieces like "redients" (ingredients) or "oliv" (olive);
    3. every token that contains a word of >= 4 letters (e.g. "cooking" in "cookingware").
    """
    words = sorted({w.lower() for w in words if w})
    if not words:
        return set()
    ids: set[int] = set()
    for w in words:
        for form in (w, w.capitalize(), w.upper()):
            for pre in ("", " "):
                ids.update(tokenizer(pre + form, add_special_tokens=False)["input_ids"])
    long_words = [w for w in words if len(w) >= 4]
    for i in range(len(tokenizer)):
        t = "".join(ch for ch in tokenizer.decode([i]).lower() if ch.isalpha())
        if len(t) >= 4 and any(t in w for w in words):
            ids.add(i)
        elif len(t) >= 4 and any(w in t for w in long_words):
            ids.add(i)
    return ids
