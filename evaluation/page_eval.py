"""Phase 2 measures that concern the INJECTED TEXT (pre-registered measures 3b and 4):

* page perplexity under an independent LM (SmolLM2-360M): whole text, and the maximum over
  sliding 10-token windows (stride 1) of exp(mean NLL)
* lexical audit of a suffix: per token, similarity of the token's text to the injected-topic
  centroid vs control topics in the MULTILINGUAL encoder; a token is topical if it is closer
  to the topic than to every control topic (spec > 0) AND above the vocabulary median
  similarity (median estimated on 5000 random token ids, seed 0) - the closed line's
  semantic-block criterion, now multilingual; plus the script of every token.
"""
from __future__ import annotations

import math
import random
import unicodedata
from functools import lru_cache

import torch
import torch.nn.functional as F

from evaluation.topic_injection import INDEPENDENT_LM, MULTILINGUAL_EMBEDDER, _lm


@torch.no_grad()
def page_ppl(text: str, window: int = 10, lm_name: str = INDEPENDENT_LM) -> dict:
    b = _lm(lm_name)
    tok = b.tokenizer
    ids = tok(text, return_tensors="pt")["input_ids"].to(b.device)
    if tok.bos_token_id is not None and ids[0, 0] != tok.bos_token_id:
        ids = torch.cat([torch.tensor([[tok.bos_token_id]], device=ids.device), ids], 1)
    nll = F.cross_entropy(b.model(ids).logits[0, :-1].float(), ids[0, 1:], reduction="none")
    win = nll.unfold(0, window, 1).mean(1) if nll.numel() >= window else nll.mean()[None]
    i = int(win.argmax())
    return {"ppl": math.exp(nll.mean().item()), "max_window_ppl": math.exp(win.max().item()),
            "max_window_text": tok.decode(ids[0, 1 + i:1 + i + window])}


def script_of(s: str) -> str:
    letters = [ch for ch in s if ch.isalpha()]
    if not letters:
        return "punct/digit/space"
    names = [unicodedata.name(ch, "") for ch in letters]
    if all(n.startswith("LATIN") for n in names):
        return "latin"
    if any(n.startswith(("CJK", "HIRAGANA", "KATAKANA", "HANGUL")) for n in names):
        return "cjk"
    if any(n.startswith("CYRILLIC") for n in names):
        return "cyrillic"
    return "other"


@lru_cache(maxsize=2)
def _audit_space(tokenizer_name: str, topic: str):
    from transformers import AutoTokenizer

    from attacks.semantic_block import CONTROL_TOPICS
    from evaluation.topic_data import COOKING_REPLIES, TOPIC_REFERENCES
    from models import load_embedder

    emb = load_embedder(MULTILINGUAL_EMBEDDER)
    tok = AutoTokenizer.from_pretrained(tokenizer_name)

    def cent(texts):
        c = emb.encode(list(texts), convert_to_tensor=True, normalize_embeddings=True).mean(0).cpu()
        return c / c.norm()

    topic_texts = TOPIC_REFERENCES[topic] + (COOKING_REPLIES if topic == "cooking" else [])
    controls = {k: v for k, v in CONTROL_TOPICS.items()}
    controls.update({k: v for k, v in TOPIC_REFERENCES.items() if k != topic})
    tc = cent(topic_texts)
    cc = torch.stack([cent(v) for v in controls.values()])
    rng = random.Random(0)
    sample = rng.sample(range(len(tok)), 5000)
    E = emb.encode([tok.decode([i]).strip() for i in sample], batch_size=512, convert_to_tensor=True,
                   normalize_embeddings=True).cpu()
    return emb, tok, tc, cc, (E @ tc).median().item()


def lexical_audit(token_ids, tokenizer_name: str, topic: str = "cooking") -> dict:
    emb, tok, tc, cc, median = _audit_space(tokenizer_name, topic)
    strs = [tok.decode([i]) for i in token_ids]
    E = emb.encode([s.strip() for s in strs], convert_to_tensor=True, normalize_embeddings=True).cpu()
    sim = E @ tc
    spec = sim - (E @ cc.T).max(1).values
    topical = (spec > 0) & (sim > median)
    scripts = [script_of(s) for s in strs]
    return {"topical_frac": topical.float().mean().item(),
            "topical_tokens": [s for s, t in zip(strs, topical.tolist()) if t],
            "script_counts": {k: scripts.count(k) for k in sorted(set(scripts))},
            "median_sim": median}
