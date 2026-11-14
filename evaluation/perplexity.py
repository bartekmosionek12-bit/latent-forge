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
