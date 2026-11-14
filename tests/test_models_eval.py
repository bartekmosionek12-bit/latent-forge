"""Smoke tests on the real toy model (distilgpt2, CPU)."""
import pytest
import torch

from evaluation import activation_projection, generate_greedy, residual_activations, target_loss
from models import ModelConfig, get_decoder_layers, load_model, split_prompt


@pytest.fixture(scope="module")
def bundle():
    return load_model(ModelConfig(name="distilgpt2"))


def test_load_and_layers(bundle):
    assert len(get_decoder_layers(bundle.model)) == 6
    assert bundle.tokenizer.pad_token is not None


def test_prompt_split_base_model(bundle):
    parts = split_prompt(bundle.tokenizer, "Hello:")
    assert parts.before == "Hello:" and parts.after == ""


def test_text_metrics(bundle):
    gen = generate_greedy(bundle, "The capital of France is", max_new_tokens=5)
    assert isinstance(gen, str) and gen
    # a continuation the model itself produced must have near-perfect token accuracy
    loss, acc = target_loss(bundle, "The capital of France is", "", gen)
    assert acc == 1.0 and loss < 5


def test_activation_projection(bundle):
    h = residual_activations(bundle, "hello world", layer=3)
    assert h.shape == (2, bundle.model.config.n_embd)
    d = h.mean(0)
    assert activation_projection(bundle, "hello world", d, 3) > 0


def test_perplexity_orders_fluent_below_gibberish(bundle):
    from evaluation import perplexity
    fluent = perplexity(bundle, "The cat sat on the mat and fell asleep in the sun.")
    junk = perplexity(bundle, "mat sun the asleep fell cat sat on and in the the.")
    assert 1 < fluent < junk
