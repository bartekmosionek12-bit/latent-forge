"""HF model + tokenizer loading. The model name is pure config: distilgpt2 on CPU
now, Qwen/Qwen2.5-7B-Instruct in fp16 on a GPU later, same code."""
from __future__ import annotations

from dataclasses import dataclass

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

_DTYPES = {"float32": torch.float32, "float16": torch.float16, "bfloat16": torch.bfloat16}


@dataclass
class ModelConfig:
    name: str = "distilgpt2"
    dtype: str = "float32"          # "float16" for Qwen2.5-7B on A5000
    device: str = "auto"            # "auto" -> cuda if available else cpu
    trust_remote_code: bool = False

    def resolved_device(self) -> str:
        if self.device != "auto":
            return self.device
        return "cuda" if torch.cuda.is_available() else "cpu"


@dataclass
class ModelBundle:
    model: torch.nn.Module
    tokenizer: object
    config: ModelConfig

    @property
    def device(self) -> torch.device:
        return next(self.model.parameters()).device


def load_model(config: ModelConfig | None = None) -> ModelBundle:
    config = config or ModelConfig()
    tok = AutoTokenizer.from_pretrained(config.name, trust_remote_code=config.trust_remote_code)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        config.name,
        torch_dtype=_DTYPES[config.dtype],
        trust_remote_code=config.trust_remote_code,
    ).to(config.resolved_device())
    model.eval()
    model.requires_grad_(False)
    return ModelBundle(model=model, tokenizer=tok, config=config)


def get_decoder_layers(model: torch.nn.Module) -> torch.nn.ModuleList:
    """Return the list of transformer blocks for common HF architectures
    (GPT-2 family: model.transformer.h; Llama/Qwen/Mistral: model.model.layers)."""
    for path in ("transformer.h", "model.layers", "gpt_neox.layers", "model.decoder.layers"):
        obj = model
        try:
            for attr in path.split("."):
                obj = getattr(obj, attr)
        except AttributeError:
            continue
        return obj
    raise ValueError(f"don't know where the decoder layers of {type(model).__name__} live")
