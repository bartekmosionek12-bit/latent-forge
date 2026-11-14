"""Success metric for ActivationDirectionTarget: projection of residual-stream
activations (output of decoder block ``layer``) onto the unit target direction."""
from __future__ import annotations

import torch

from models import get_decoder_layers


@torch.no_grad()
def residual_activations(bundle, text: str, layer: int) -> torch.Tensor:
    """[seq, d_model] output of decoder block ``layer`` for ``text``."""
    captured = {}

    def hook(_mod, _inp, out):
        captured["h"] = (out[0] if isinstance(out, tuple) else out).detach()

    handle = get_decoder_layers(bundle.model)[layer].register_forward_hook(hook)
    try:
        ids = bundle.tokenizer(text, return_tensors="pt")["input_ids"].to(bundle.device)
        bundle.model(ids)
    finally:
        handle.remove()
    return captured["h"][0].float()


#: user turn used when a bare text is placed as the assistant reply (chat mode)
DEFAULT_CHAT_USER = "Write a short message."


def has_real_chat_template(tokenizer) -> bool:
    """True for a model-provided template; False for none or nanoGCG's fallback
    (nanogcg installs a plain-concatenation template on base-model tokenizers)."""
    from models.prompting import NANOGCG_FALLBACK_TEMPLATE

    t = getattr(tokenizer, "chat_template", None)
    return bool(t) and t != NANOGCG_FALLBACK_TEMPLATE


@torch.no_grad()
def content_activations(bundle, text: str, layer: int, chat: bool = False,
                        user: str = DEFAULT_CHAT_USER, skip_first: bool = True) -> torch.Tensor:
    """Block-``layer`` activations for the tokens of ``text`` only.

    chat=False: raw text; position 0 dropped if ``skip_first`` (GPT-2 attention sink).
    chat=True : ``text`` is rendered as the ASSISTANT reply to ``user`` with the
    model's chat template; only tokens whose character span lies inside ``text``
    are kept (template / system / user tokens are excluded via offset mapping).
    """
    if not chat:
        h = residual_activations(bundle, text, layer)
        return h[1:] if skip_first and h.shape[0] > 1 else h

    tok = bundle.tokenizer
    rendered = tok.apply_chat_template(
        [{"role": "user", "content": user}, {"role": "assistant", "content": text}], tokenize=False)
    start = rendered.rindex(text)
    end = start + len(text)
    enc = tok(rendered, add_special_tokens=False, return_offsets_mapping=True, return_tensors="pt")
    offs = enc["offset_mapping"][0]
    keep = (offs[:, 0] >= start) & (offs[:, 1] <= end) & (offs[:, 1] > offs[:, 0])
    if not keep.any():
        raise ValueError("no content tokens found in rendered chat")
    captured = {}

    def hook(_mod, _inp, out):
        captured["h"] = (out[0] if isinstance(out, tuple) else out).detach()

    handle = get_decoder_layers(bundle.model)[layer].register_forward_hook(hook)
    try:
        bundle.model(enc["input_ids"].to(bundle.device))
    finally:
        handle.remove()
    return captured["h"][0].float()[keep.to(captured["h"].device)]


def activation_projection(bundle, text: str, direction: torch.Tensor, layer: int,
                          skip_first: bool = True, chat: bool = False,
                          user: str = DEFAULT_CHAT_USER) -> float:
    """Mean over the text's tokens of <h_t, direction/|direction|>. In raw mode
    position 0 is skipped by default: in GPT-2 it carries a huge-norm
    "attention sink" activation that dominates any mean."""
    h = content_activations(bundle, text, layer, chat=chat, user=user, skip_first=skip_first)
    d = direction.float().to(h.device)
    return (h @ (d / d.norm())).mean().item()
