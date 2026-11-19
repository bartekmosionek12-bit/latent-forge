"""Open prompt-injection classifiers used as realistic detectors (pre-registered measure 3a).

D1 = protectai/deberta-v3-base-prompt-injection-v2 (Apache-2.0)  -> the D arm optimises against it
D2 = deepset/deberta-v3-base-injection (MIT)                       -> held out, never seen by the optimiser
Both are public, not gated; no token is needed. Input = the page text as the app would
retrieve it (page + pasted attack). Score = P(label "INJECTION")."""
from __future__ import annotations

from functools import lru_cache

import torch

D1 = "protectai/deberta-v3-base-prompt-injection-v2"
D2 = "deepset/deberta-v3-base-injection"


@lru_cache(maxsize=2)
def _load(name: str, device: str):
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(name)
    model = AutoModelForSequenceClassification.from_pretrained(name).to(device).eval()
    labels = {i: l.upper() for i, l in model.config.id2label.items()}
    inj = [i for i, l in labels.items() if "INJECTION" in l]
    if len(inj) != 1:
        raise ValueError(f"{name}: cannot find the injection label in {labels}")
    return tok, model, inj[0]


@torch.no_grad()
def injection_prob(texts, name: str = D1, device: str = "cpu", batch: int = 16) -> list[float]:
    tok, model, k = _load(name, device)
    out = []
    for i in range(0, len(texts), batch):
        enc = tok(list(texts[i:i + batch]), truncation=True, max_length=512, padding=True,
                  return_tensors="pt").to(device)
        out += model(**enc).logits.float().softmax(-1)[:, k].tolist()
    return out
