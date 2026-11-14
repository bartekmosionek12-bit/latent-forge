"""ActivationSteeringAttack: contrastive direction in the attacked model's residual
stream, injected via forward hooks during generation (activation addition,
cf. "Steering in the Shadows", arXiv 2511.17194).

direction = mean_p(mean_t h_L(positive_p)) - mean_n(mean_t h_L(negative_n))
where h_L is the output of decoder block ``layer`` (plain PyTorch forward hook)
and token position 0 is excluded (GPT-2 attention-sink activation).
Injection: block-L output h -> h + alpha * direction at every position, every
forward pass of generate(). ``alpha`` is the strength knob (alpha=0 = no steering).

Hook semantics verified on transformers 4.47.1 / torch 2.15: GPT2Block returns a
tuple (hidden_states, present, ...); returning a modified tuple from
register_forward_hook replaces the block output; removing the handle restores
the original forward exactly.
"""
from __future__ import annotations

import time
from contextlib import contextmanager

import torch

from evaluation.activation import residual_activations
from models import get_decoder_layers
from targets import ActivationDirectionTarget

from .base import Attack, AttackResult

DEFAULTS = dict(
    alphas=(0.0, 2.0, 4.0, 8.0),
    test_prompts=("Thanks for the update about the meeting.",),
    max_new_tokens=30,
    skip_first_token=True,
    repetition_penalty=1.3,   # same for every alpha; stops distilgpt2 greedy loops
)


@contextmanager
def steer(bundle, layer: int, vector: torch.Tensor):
    """Add ``vector`` to the output of decoder block ``layer`` while active."""
    vec = vector.to(bundle.device, next(bundle.model.parameters()).dtype)

    def hook(_mod, _args, out):
        if isinstance(out, tuple):
            return (out[0] + vec,) + out[1:]
        return out + vec

    handle = get_decoder_layers(bundle.model)[layer].register_forward_hook(hook)
    try:
        yield
    finally:
        handle.remove()


def contrastive_direction(bundle, target: ActivationDirectionTarget, skip_first: bool = True):
    def mean_act(prompts):
        acts = []
        for p in prompts:
            h = residual_activations(bundle, p, target.layer)
            acts.append((h[1:] if skip_first and h.shape[0] > 1 else h).mean(0))
        return torch.stack(acts).mean(0)

    pos, neg = mean_act(target.positive_prompts), mean_act(target.negative_prompts)
    return pos - neg, pos, neg


@torch.no_grad()
def generate(bundle, prompt: str, max_new_tokens: int, repetition_penalty: float = 1.0) -> str:
    tok = bundle.tokenizer
    ids = tok(prompt, return_tensors="pt")["input_ids"].to(bundle.device)
    out = bundle.model.generate(ids, attention_mask=torch.ones_like(ids), do_sample=False,
                                max_new_tokens=max_new_tokens, pad_token_id=tok.pad_token_id,
                                repetition_penalty=repetition_penalty)
    return tok.decode(out[0, ids.shape[1]:], skip_special_tokens=True)


class ActivationSteeringAttack(Attack):
    target_type = ActivationDirectionTarget

    def __init__(self, target: ActivationDirectionTarget, **config) -> None:
        unknown = set(config) - set(DEFAULTS)
        if unknown:
            raise TypeError(f"unknown ActivationSteeringAttack options: {sorted(unknown)}")
        super().__init__(target, **{**DEFAULTS, **config})

    def run(self, bundle) -> AttackResult:
        c, t0 = self.config, time.time()
        n_layers = len(get_decoder_layers(bundle.model))
        if not 0 <= self.target.layer < n_layers:
            raise ValueError(f"layer {self.target.layer} out of range (model has {n_layers})")
        direction, pos_mean, neg_mean = contrastive_direction(bundle, self.target, c["skip_first_token"])

        generations: dict[float, dict[str, str]] = {}
        for alpha in c["alphas"]:
            generations[alpha] = {}
            with steer(bundle, self.target.layer, alpha * direction):
                for p in c["test_prompts"]:
                    generations[alpha][p] = generate(bundle, p, c["max_new_tokens"], c["repetition_penalty"])

        return AttackResult(
            target=self.target, attack=type(self).__name__, payload=direction.cpu(),
            final_loss=None,
            extra={"generations": generations,
                   "direction_norm": direction.norm().item(),
                   "pos_mean_norm": pos_mean.norm().item(),
                   "neg_mean_norm": neg_mean.norm().item(),
                   "seconds": time.time() - t0,
                   "config": {k: list(v) if isinstance(v, tuple) else v for k, v in c.items()}},
        )
