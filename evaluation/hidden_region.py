"""Evaluation for HiddenStateRegionTarget: for each suffix variant, measure
(1) the optimisation metric - cosine distance of the last-token block-L state of the
    JOINTLY tokenised string [prompt + suffix] to the region centroid, and
(2) the independent topical score of the greedy continuation (evaluation/topical.py).
Controls: no suffix, initial filler, random-token suffixes, naive topical suffix."""
from __future__ import annotations

import torch

from models import get_decoder_layers, split_prompt

from .topical import COOKING_KEYWORDS, topical_scores

GEN_KW = dict(max_new_tokens=40, repetition_penalty=1.3)


@torch.no_grad()
def last_token_distance(bundle, text: str, layer: int, centroid: torch.Tensor) -> float:
    ids = bundle.tokenizer(text, return_tensors="pt")["input_ids"].to(bundle.device)
    cap = {}
    blk = get_decoder_layers(bundle.model)[layer]
    h = blk.register_forward_hook(lambda _m, _a, o: cap.__setitem__("h", o[0] if isinstance(o, tuple) else o))
    try:
        bundle.model(ids)
    finally:
        h.remove()
    v = cap["h"][0, -1].float()
    return 1.0 - torch.nn.functional.cosine_similarity(v, centroid.float().to(v.device), dim=0).item()


@torch.no_grad()
def continuation(bundle, text: str) -> str:
    tok = bundle.tokenizer
    ids = tok(text, return_tensors="pt")["input_ids"].to(bundle.device)
    out = bundle.model.generate(ids, attention_mask=torch.ones_like(ids), do_sample=False,
                                pad_token_id=tok.pad_token_id, **GEN_KW)
    return tok.decode(out[0, ids.shape[1]:], skip_special_tokens=True)


def random_suffixes(bundle, n_tokens: int, seeds=(0, 1, 2, 3, 4)) -> list[str]:
    from attacks.gcg_region import allowed_token_mask

    tok = bundle.tokenizer
    allowed = allowed_token_mask(tok, bundle.model.get_input_embeddings().weight.shape[0]).nonzero()[:, 0]
    out = []
    for s in seeds:
        g = torch.Generator().manual_seed(s)
        out.append(tok.decode(allowed[torch.randint(0, len(allowed), (n_tokens,), generator=g)]))
    return out


def naive_topical_suffix(bundle, reference_texts, n_tokens: int) -> str:
    """Reference sentences concatenated and cut to the same token budget."""
    tok = bundle.tokenizer
    ids = tok(" " + " ".join(reference_texts), add_special_tokens=False)["input_ids"][:n_tokens]
    return tok.decode(ids)


def evaluate_hidden_region(result, bundle) -> dict:
    t = result.target
    c = t.centroid(bundle)
    parts = split_prompt(bundle.tokenizer, t.prompt)
    kw = t.topic_keywords or COOKING_KEYWORDS
    n = result.extra["config"]["n_tokens"]

    def score(suffix: str) -> dict:
        text = parts.before + suffix + parts.after
        cont = continuation(bundle, text)
        return {"suffix": suffix,
                "hidden_cos_dist": last_token_distance(bundle, text, t.layer, c),
                "continuation": cont,
                **topical_scores(cont, t.reference_texts, t.eval_embedder_name, kw)}

    variants = {"no_suffix": score(""), "init_filler": score(result.extra["init_suffix"])}
    for i, s in enumerate(random_suffixes(bundle, n)):
        variants[f"random_{i}"] = score(s)
    variants["naive_topical"] = score(naive_topical_suffix(bundle, t.reference_texts, n))
    variants["optimised"] = score(result.payload)

    refs = {"reference_last_token_hidden_cos_dist":
            [last_token_distance(bundle, r, t.layer, c) for r in t.reference_texts],
            "reference_minilm_cos_dist_to_topic":
            [topical_scores(r, t.reference_texts, t.eval_embedder_name, kw)["minilm_cos_dist_to_topic"]
             for r in t.reference_texts]}
    return {"variants": variants, "reference_scale": refs}
