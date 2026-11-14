"""Builds the exact token layout nanoGCG optimises over, so evaluation sees the
same tokens the attack saw: [before] [optim/suffix] [after] -> generation.

Mirrors nanogcg 0.3.0 GCG.run(): user message = prompt + "{optim_str}",
apply_chat_template(add_generation_prompt=True), strip BOS, split on the
placeholder. Base models without a chat template get nanogcg's fallback
template (plain concatenation of message contents)."""
from __future__ import annotations

from dataclasses import dataclass

import torch

NANOGCG_FALLBACK_TEMPLATE = "{% for message in messages %}{{ message['content'] }}{% endfor %}"


@dataclass
class PromptParts:
    before: str
    after: str

    def input_ids(self, tokenizer, suffix: str, device) -> torch.Tensor:
        """Tokenise segment-wise (as nanogcg does) and concatenate."""
        before = tokenizer([self.before], padding=False, return_tensors="pt")["input_ids"]
        mid = tokenizer([suffix], add_special_tokens=False, return_tensors="pt")["input_ids"]
        after = tokenizer([self.after], add_special_tokens=False, return_tensors="pt")["input_ids"]
        # empty strings tokenise to float tensors of shape [1, 0]; force int64
        return torch.cat([before, mid, after], dim=1).to(device, torch.int64)


def ensure_chat_template(tokenizer) -> None:
    if not getattr(tokenizer, "chat_template", None):
        tokenizer.chat_template = NANOGCG_FALLBACK_TEMPLATE


def split_prompt(tokenizer, prompt: str) -> PromptParts:
    ensure_chat_template(tokenizer)
    messages = [{"role": "user", "content": prompt + "{optim_str}"}]
    template = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    if tokenizer.bos_token and template.startswith(tokenizer.bos_token):
        template = template.replace(tokenizer.bos_token, "")
    before, after = template.split("{optim_str}")
    return PromptParts(before=before, after=after)
