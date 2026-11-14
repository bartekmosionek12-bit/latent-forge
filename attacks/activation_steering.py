"""ActivationSteeringAttack: contrastive direction in the attacked model's residual
stream, injected via forward hooks during generation (activation addition,
cf. "Steering in the Shadows", arXiv 2511.17194).

direction = mean_p(mean_t h_L(positive_p)) - mean_n(mean_t h_L(negative_n))
where h_L is the output of decoder block ``layer`` (plain PyTorch forward hook)
and token position 0 is excluded (GPT-2 attention-sink activation).
Injection: block-L output h -> h + alpha * direction at every position, every
forward pass of generate(). ``alpha`` is the strength knob (alpha=0 = no steering).

Chat models (chat_template=True, or "auto" when the tokenizer ships a real
template): contrastive texts are rendered as the ASSISTANT reply in the chat
template and only their own tokens are averaged; test prompts are user turns and
generation starts the assistant turn. Base models keep the raw-text behaviour.

Hook semantics verified on transformers 4.47.1 / torch 2.15: GPT2Block returns a
tuple (hidden_states, present, ...); returning a modified tuple from
register_forward_hook replaces the block output; removing the handle restores
the original forward exactly.
"""
from __future__ import annotations

import time
from contextlib import contextmanager

import torch

from evaluation.activation import content_activations, has_real_chat_template
from models import get_decoder_layers
from targets import ActivationDirectionTarget

from .base import Attack, AttackResult

DEFAULTS = dict(
    alphas=(0.0, 2.0, 4.0, 8.0),
    test_prompts=("Thanks for the update about the meeting.",),
    max_new_tokens=30,
    skip_first_token=True,
    # same for every alpha. None = auto: 1.3 for raw-text/base models (stops
    # distilgpt2 greedy loops), 1.0 in chat mode (1.3 distorts instruct baselines,
    # observed on Qwen2.5-0.5B-Instruct: it penalises tokens of the system/user turn)
    repetition_penalty=None,
    chat_template="auto",     # True/False/"auto" (= model ships a real chat template)
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


def contrastive_direction(bundle, target: ActivationDirectionTarget, skip_first: bool = True,
                          chat: bool = False):
    """chat=True: each contrastive text is rendered as the assistant reply in the
    model's chat template and only its own tokens are averaged."""
    def mean_act(prompts):
        acts = [content_activations(bundle, p, target.layer, chat=chat, skip_first=skip_first).mean(0)
                for p in prompts]
        return torch.stack(acts).mean(0)

    pos, neg = mean_act(target.positive_prompts), mean_act(target.negative_prompts)
    return pos - neg, pos, neg


@torch.no_grad()
def generate(bundle, prompt: str, max_new_tokens: int, repetition_penalty: float = 1.0,
             chat: bool = False) -> str:
    """chat=True: ``prompt`` is the user turn, generation starts the assistant turn."""
    tok = bundle.tokenizer
    if chat:
        ids = tok.apply_chat_template([{"role": "user", "content": prompt}],
                                      add_generation_prompt=True, return_tensors="pt").to(bundle.device)
    else:
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
        chat = c["chat_template"]
        if chat == "auto":
            chat = has_real_chat_template(bundle.tokenizer)
        rp = c["repetition_penalty"]
        if rp is None:
            rp = 1.0 if chat else 1.3
        direction, pos_mean, neg_mean = contrastive_direction(bundle, self.target,
                                                              c["skip_first_token"], chat)

        generations: dict[float, dict[str, str]] = {}
        for alpha in c["alphas"]:
            generations[alpha] = {}
            with steer(bundle, self.target.layer, alpha * direction):
                for p in c["test_prompts"]:
                    generations[alpha][p] = generate(bundle, p, c["max_new_tokens"], rp, chat)

        return AttackResult(
            target=self.target, attack=type(self).__name__, payload=direction.cpu(),
            final_loss=None,
            extra={"generations": generations, "chat_template": chat, "repetition_penalty": rp,
                   "direction_norm": direction.norm().item(),
                   "pos_mean_norm": pos_mean.norm().item(),
                   "neg_mean_norm": neg_mean.norm().item(),
                   "seconds": time.time() - t0,
                   "config": {k: list(v) if isinstance(v, tuple) else v for k, v in c.items()}},
        )
