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


def activation_projection(bundle, text: str, direction: torch.Tensor, layer: int,
                          skip_first: bool = True) -> float:
    """Mean over tokens of <h_t, direction/|direction|>. Position 0 is skipped by
    default: in GPT-2 it carries a huge-norm "attention sink" activation that
    dominates any mean."""
    h = residual_activations(bundle, text, layer)
    if skip_first and h.shape[0] > 1:
        h = h[1:]
    d = direction.float().to(h.device)
    return (h @ (d / d.norm())).mean().item()
