"""Perplexity of a text under a (separate, small) causal LM: exp(mean token NLL).
Used as a detectability signal for adversarial passages, not as an attack term."""
from __future__ import annotations

import math

import torch
import torch.nn.functional as F


@torch.no_grad()
def perplexity(bundle, text: str) -> float:
    tok = bundle.tokenizer
    ids = tok(text, return_tensors="pt")["input_ids"].to(bundle.device)
    if tok.bos_token_id is not None:  # condition the first real token on BOS
        ids = torch.cat([torch.tensor([[tok.bos_token_id]], device=ids.device), ids], dim=1)
    logits = bundle.model(ids).logits[0, :-1].float()
    return math.exp(F.cross_entropy(logits, ids[0, 1:]).item())


@torch.no_grad()
def conditional_nll(bundle, prefix: str, text: str) -> float:
    """Mean NLL (nats/token) of ``text``'s tokens given ``prefix`` (segment-wise
    tokenisation and no BOS, exactly as the attack builds its inputs, so for the attacked
    model this equals the attack's own CE_suffix). exp() of it is the suffix PPL."""
    tok = bundle.tokenizer
    p = tok(prefix, add_special_tokens=False)["input_ids"]
    t = tok(text, add_special_tokens=False)["input_ids"]
    ids = torch.tensor([p + t], device=bundle.device)
    logits = bundle.model(ids).logits[0, len(p) - 1:-1].float()
    return F.cross_entropy(logits, ids[0, len(p):]).item()
