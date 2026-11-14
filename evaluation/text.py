"""Success metrics for TextTarget."""
from __future__ import annotations

import torch
import torch.nn.functional as F

from models import split_prompt


@torch.no_grad()
def generate_greedy(bundle, prompt: str, suffix: str = "", max_new_tokens: int = 20) -> str:
    tok = bundle.tokenizer
    ids = split_prompt(tok, prompt).input_ids(tok, suffix, bundle.device)
    out = bundle.model.generate(ids, attention_mask=torch.ones_like(ids), do_sample=False,
                                max_new_tokens=max_new_tokens, pad_token_id=tok.pad_token_id)
    return tok.decode(out[0, ids.shape[1]:], skip_special_tokens=True)


@torch.no_grad()
def target_loss(bundle, prompt: str, suffix: str, target: str) -> tuple[float, float]:
    """Teacher-forced mean CE of the target tokens (same quantity nanoGCG minimises)
    and position-wise greedy accuracy on the target tokens."""
    tok = bundle.tokenizer
    ids = split_prompt(tok, prompt).input_ids(tok, suffix, bundle.device)
    tgt = tok([target], add_special_tokens=False, return_tensors="pt")["input_ids"].to(bundle.device)
    full = torch.cat([ids, tgt], dim=1)
    logits = bundle.model(full).logits[0, ids.shape[1] - 1: -1].float()
    loss = F.cross_entropy(logits, tgt[0]).item()
    acc = (logits.argmax(-1) == tgt[0]).float().mean().item()
    return loss, acc


def evaluate_text_target(bundle, target, suffix: str) -> dict:
    n_tgt = len(bundle.tokenizer(target.text, add_special_tokens=False)["input_ids"])
    report = {}
    for label, sfx in (("without_suffix", ""), ("with_suffix", suffix)):
        gen = generate_greedy(bundle, target.prompt, sfx, max_new_tokens=n_tgt + 8)
        loss, acc = target_loss(bundle, target.prompt, sfx, target.text)
        report[label] = {
            "generation": gen,
            "exact_match": gen.startswith(target.text),
            "target_token_accuracy": acc,
            "target_loss": loss,
        }
    return report
