"""EmbeddingRegionAttack (basic version): push a text into a region of an
EXTERNAL embedder's space (RAG / memory retriever), white-box.

Passage layout:   [payload text] + " " + [optimised span of n tokens]
  * payload: fixed, attacker-chosen content that should get retrieved
  * span: initialised by concatenating the "centroid text" (the reference text
    closest to the region centroid, truncated/padded to n tokens) or neutral
    filler, then refined by gradient-guided token substitution
    (HotFlip / GCG-style coordinate search, as in SentAttack-like corpus poisoning).

Objective: minimise cosine distance 1 - cos(E(passage), centroid).

Differentiable path: the real SentenceTransformer pipeline
(Transformer -> Pooling -> Normalize) called with ``inputs_embeds`` instead of
``input_ids``; verified to match ``encode()`` exactly for all-MiniLM-L6-v2 with
sentence-transformers 5.7.0. Candidates are scored with exact forward passes;
the final distance is measured on the decoded STRING via ``encode()``.

Not in this version (later steps): coherence/perplexity regulariser,
black-box surrogate embedder, region-radius experiments.
"""
from __future__ import annotations

import time

import torch

from models import load_embedder
from targets import EmbeddingRegionTarget

from .base import Attack, AttackResult

DEFAULTS = dict(
    payload="",                # fixed text the span is appended to
    n_tokens=16,               # length of the optimised span
    init="centroid_text",      # "centroid_text" | "neutral"
    num_steps=100,
    topk=64,                   # top-k replacement tokens per position from the gradient
    search_width=128,          # candidates evaluated per step (1 position flipped each)
    seed=0,
    device="auto",
)


class EmbeddingRegionAttack(Attack):
    target_type = EmbeddingRegionTarget

    def __init__(self, target: EmbeddingRegionTarget, **config) -> None:
        unknown = set(config) - set(DEFAULTS)
        if unknown:
            raise TypeError(f"unknown EmbeddingRegionAttack options: {sorted(unknown)}")
        super().__init__(target, **{**DEFAULTS, **config})

    # ---- helpers -------------------------------------------------------------
    def _setup(self):
        c = self.config
        self.embedder = load_embedder(self.target.embedder_name, c["device"])
        self.tok = self.embedder.tokenizer
        self.W = self.embedder[0].auto_model.get_input_embeddings().weight  # [V, d]
        self.dev = self.W.device
        self.centroid = self.target.centroid(self.embedder).to(self.dev)
        self.allowed = self._allowed_token_mask()

    def _allowed_token_mask(self) -> torch.Tensor:
        """Whole-vocab mask: no special tokens, no [unusedN], ASCII only."""
        mask = torch.zeros(self.W.shape[0], dtype=torch.bool)
        special = set(self.tok.all_special_ids)
        for tok_str, idx in self.tok.get_vocab().items():
            if idx in special or tok_str.startswith("[") or not tok_str.isascii():
                continue
            if not any(ch.isalnum() for ch in tok_str):
                continue
            mask[idx] = True
        return mask.to(self.dev)

    def _encode_ids(self, text: str) -> torch.Tensor:
        return self.tok(text, add_special_tokens=False, return_tensors="pt")["input_ids"][0].to(self.dev)

    def _passage_ids(self, span: torch.Tensor) -> torch.Tensor:
        """[CLS] payload span [SEP] as a 1-D id tensor."""
        parts = [torch.tensor([self.tok.cls_token_id], device=self.dev)]
        if self.payload_ids.numel():
            parts.append(self.payload_ids)
        parts += [span, torch.tensor([self.tok.sep_token_id], device=self.dev)]
        return torch.cat(parts)

    def _embed_from_ids(self, ids: torch.Tensor) -> torch.Tensor:
        """Batched exact embeddings for [B, L] id tensors."""
        feats = {"input_ids": ids, "attention_mask": torch.ones_like(ids),
                 "token_type_ids": torch.zeros_like(ids)}
        return self.embedder(feats)["sentence_embedding"]

    def _loss_ids(self, ids: torch.Tensor) -> torch.Tensor:
        return 1 - self._embed_from_ids(ids) @ self.centroid

    def _span_grad(self, span: torch.Tensor) -> torch.Tensor:
        """d loss / d one_hot(span) -> [n, V]."""
        ids = self._passage_ids(span).unsqueeze(0)
        one_hot = torch.nn.functional.one_hot(span, self.W.shape[0]).to(self.W.dtype)
        one_hot.requires_grad_(True)
        embeds = self.W[ids[0]].detach().clone()
        start = 1 + self.payload_ids.numel()
        embeds = torch.cat([embeds[:start], one_hot @ self.W, embeds[start + span.numel():]])
        feats = {"inputs_embeds": embeds.unsqueeze(0), "attention_mask": torch.ones_like(ids),
                 "token_type_ids": torch.zeros_like(ids)}
        e = self.embedder(feats)["sentence_embedding"][0]
        (1 - e @ self.centroid).backward()
        return one_hot.grad

    def _roundtrip_ok(self, span: torch.Tensor) -> bool:
        """The span must survive decode -> re-encode unchanged (we ship a string)."""
        text = self.tok.decode(span)
        again = self._encode_ids(text)
        return again.numel() == span.numel() and bool((again == span).all())

    def _init_span(self) -> tuple[torch.Tensor, str]:
        n, c = self.config["n_tokens"], self.config
        filler = self._encode_ids("the")
        if c["init"] == "centroid_text":
            refs = self.target.reference_texts
            embs = self.embedder.encode(refs, convert_to_tensor=True, normalize_embeddings=True)
            src = refs[int((embs.to(self.dev) @ self.centroid).argmax())]
            ids = self._encode_ids(src)[:n]
        elif c["init"] == "neutral":
            src, ids = "the " * n, filler.repeat(n)
        else:
            raise ValueError(f"unknown init {c['init']!r}")
        if ids.numel() < n:
            ids = torch.cat([ids, filler.repeat(n - ids.numel())])
        return ids, src

    def passage_text(self, span: torch.Tensor) -> str:
        span_text = self.tok.decode(span)
        return f"{self.config['payload']} {span_text}".strip()

    # ---- main loop -----------------------------------------------------------
    def run(self, bundle=None) -> AttackResult:
        """``bundle`` (the attacked LLM) is unused: the target lives in the embedder."""
        c = self.config
        torch.manual_seed(c["seed"])
        self._setup()
        self.payload_ids = self._encode_ids(c["payload"]) if c["payload"] else \
            torch.empty(0, dtype=torch.long, device=self.dev)
        span, init_source = self._init_span()
        t0 = time.time()

        with torch.no_grad():
            cur = self._loss_ids(self._passage_ids(span).unsqueeze(0)).item()
        history, accepted = [cur], 0

        for _ in range(c["num_steps"]):
            grad = self._span_grad(span)                               # [n, V]
            grad[:, ~self.allowed] = float("inf")
            top = (-grad).topk(c["topk"], dim=1).indices               # [n, k]
            pos = torch.randint(0, span.numel(), (c["search_width"],), device=self.dev)
            new_tok = top[pos, torch.randint(0, c["topk"], (c["search_width"],), device=self.dev)]
            cands = span.repeat(c["search_width"], 1)
            cands[torch.arange(c["search_width"]), pos] = new_tok
            cands = torch.unique(cands, dim=0)
            cands = cands[(cands != span).any(dim=1)]
            with torch.no_grad():
                full = torch.stack([self._passage_ids(s) for s in cands])
                losses = self._loss_ids(full)
            for i in losses.argsort().tolist():                        # best round-trippable
                if losses[i].item() >= cur:
                    break
                if self._roundtrip_ok(cands[i]):
                    span, cur, accepted = cands[i], losses[i].item(), accepted + 1
                    break
            history.append(cur)

        text = self.passage_text(span)
        with torch.no_grad():
            measured = 1 - (self.embedder.encode([text], convert_to_tensor=True,
                                                 normalize_embeddings=True)[0].to(self.dev)
                            @ self.centroid).item()
        return AttackResult(
            target=self.target, attack=type(self).__name__, payload=text,
            final_loss=measured, loss_history=history,
            extra={"payload_text": c["payload"], "init_source": init_source,
                   "init_span_text": self.tok.decode(self._init_span()[0]),
                   "span_text": self.tok.decode(span), "accepted_flips": accepted,
                   "seconds": time.time() - t0,
                   "config": dict(c)},
        )
