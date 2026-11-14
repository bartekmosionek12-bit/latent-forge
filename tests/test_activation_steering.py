"""ActivationSteeringAttack on distilgpt2 (CPU, a few seconds)."""
import pytest
import torch

import dispatcher
from attacks import ActivationSteeringAttack
from attacks.activation_steering import generate, steer
from models import ModelConfig, load_model
from targets import ActivationDirectionTarget, EmbeddingRegionTarget

POS = ["Dear Sir, I am writing to inform you of the decision.", "Kindly confirm your attendance."]
NEG = ["hey lol whats up", "thx dude see ya"]


@pytest.fixture(scope="module")
def bundle():
    return load_model(ModelConfig(name="distilgpt2"))


def test_rejects_unknown_option_and_bad_layer(bundle):
    with pytest.raises(TypeError):
        ActivationSteeringAttack(ActivationDirectionTarget(POS, NEG, 1), nope=1)
    with pytest.raises(ValueError):
        ActivationSteeringAttack(ActivationDirectionTarget(POS, NEG, 99)).run(bundle)


def test_steer_hook_is_removed(bundle):
    ids = bundle.tokenizer("Hello there", return_tensors="pt").input_ids
    base = bundle.model(ids).logits
    # (a constant vector would be a no-op: the next LayerNorm subtracts the mean)
    with steer(bundle, 2, torch.randn(768, generator=torch.Generator().manual_seed(0)) * 20):
        steered = bundle.model(ids).logits
    after = bundle.model(ids).logits
    assert (steered - base).abs().max() > 1e-2
    assert torch.equal(after, base)


def test_dispatch_alpha0_is_baseline_and_alpha_moves_projection(bundle):
    target = ActivationDirectionTarget(POS, NEG, layer=3)
    prompt = "Regarding the schedule for next week,"
    result, report = dispatcher.run(target, bundle, alphas=[0.0, 2.0], test_prompts=[prompt],
                                    max_new_tokens=12)
    assert result.attack == "ActivationSteeringAttack"
    assert result.extra["generations"][0.0][prompt] == generate(bundle, prompt, 12, 1.3)
    by = report["by_alpha"]
    assert by[2.0]["mean_projection_of_output"] > by[0.0]["mean_projection_of_output"]


def test_all_three_target_types_route():
    assert type(dispatcher.select_attack(ActivationDirectionTarget(POS, NEG, 1))).__name__ \
        == "ActivationSteeringAttack"
    assert type(dispatcher.select_attack(EmbeddingRegionTarget(POS))).__name__ \
        == "EmbeddingRegionAttack"


@pytest.mark.slow
def test_chat_mode_on_qwen_small():
    """Chat-template path on a real instruct model (Qwen2.5-0.5B-Instruct, CPU)."""
    from evaluation.activation import content_activations, has_real_chat_template

    qb = load_model(ModelConfig(name="Qwen/Qwen2.5-0.5B-Instruct"))
    assert has_real_chat_template(qb.tokenizer)
    text = "hey guys, so the meeting got moved lol"
    h = content_activations(qb, text, 12, chat=True)
    assert h.shape == (len(qb.tokenizer(text)["input_ids"]), qb.model.config.hidden_size)
    target = ActivationDirectionTarget(POS, NEG, layer=12)
    result, report = dispatcher.run(target, qb, alphas=[0.0], max_new_tokens=5,
                                    test_prompts=["Say hello."])
    assert result.extra["chat_template"] is True and report["chat_template"] is True


def test_nanogcg_fallback_template_is_not_real(bundle):
    from evaluation.activation import has_real_chat_template
    from models.prompting import ensure_chat_template

    ensure_chat_template(bundle.tokenizer)  # what nanogcg does to base models
    assert not has_real_chat_template(bundle.tokenizer)
