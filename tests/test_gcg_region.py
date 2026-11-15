"""GCGRegionAttack + independent topical evaluation on distilgpt2 (CPU)."""
import pytest
import torch

import dispatcher
from attacks import GCGRegionAttack
from evaluation.topical import COOKING_KEYWORDS, keyword_score
from models import ModelConfig, load_model
from targets import HiddenStateRegionTarget

REFS = ["Chop the onions and fry them in oil.", "Simmer the tomato sauce with basil."]


@pytest.fixture(scope="module")
def bundle():
    return load_model(ModelConfig(name="distilgpt2"))


def test_routes_and_rejects_unknown_option():
    t = HiddenStateRegionTarget(REFS, layer=3)
    assert type(dispatcher.select_attack(t)) is GCGRegionAttack
    with pytest.raises(TypeError):
        GCGRegionAttack(t, nope=1)


def test_keyword_score_is_independent_of_model():
    s = keyword_score("Fry the onions, then add salt. The council met.", COOKING_KEYWORDS)
    assert s["keyword_hits"] == ["fry", "onions", "salt"] and 0 < s["keyword_frac"] < 1


def test_ids_and_onehot_paths_agree_and_gradient_matches_fd(bundle):
    a = GCGRegionAttack(HiddenStateRegionTarget(REFS, layer=3), n_tokens=6)
    a.setup(bundle)
    s = a.init_suffix()
    e = a.W[s].detach().clone().requires_grad_(True)
    loss = a.loss_from_suffix_embeds(e)
    assert loss.item() == pytest.approx(a.loss_ids(a.full_ids(s)[None]).item(), abs=1e-5)
    g = torch.autograd.grad(loss, e)[0]
    v = g / g.norm()  # steepest direction: largest signal vs float32 noise (fd should be |g|)
    with torch.no_grad():
        fd = (a.loss_from_suffix_embeds(e + 1e-2 * v) - a.loss_from_suffix_embeds(e - 1e-2 * v)) / 2e-2
    assert fd.item() == pytest.approx((g * v).sum().item(), rel=0.05)


@pytest.mark.slow
def test_tiny_run_lowers_hidden_distance_and_reports_both_measures(bundle):
    t = HiddenStateRegionTarget(REFS, layer=3)
    result, report = dispatcher.run(t, bundle, n_tokens=6, num_steps=4, search_width=32, topk=32)
    assert result.attack == "GCGRegionAttack"
    assert result.loss_history[-1] < result.loss_history[0]
    assert result.loss_history == sorted(result.loss_history, reverse=True)
    v = report["variants"]["optimised"]
    assert {"hidden_cos_dist", "minilm_cos_dist_to_topic", "keyword_frac", "continuation"} <= set(v)
    # shipped as a string: re-tokenised hidden distance equals the attack's own number
    assert v["hidden_cos_dist"] == pytest.approx(result.final_loss, abs=1e-4)


def test_blocked_tokens_cover_subwords_and_spare_neutral(bundle):
    from attacks.gcg_region import blocked_token_ids

    ids = blocked_token_ids(bundle.tokenizer, ["ingredients", "olive", "recipe"])
    tok = bundle.tokenizer
    for piece in ["redients", " Olive", " recipe"]:
        assert all(i in ids for i in tok(piece, add_special_tokens=False)["input_ids"])
    for neutral in [" council", " the", " x"]:
        assert not any(i in ids for i in tok(neutral, add_special_tokens=False)["input_ids"])


def test_band_objective_uses_reference_midpoint(bundle):
    a = GCGRegionAttack(HiddenStateRegionTarget(REFS, layer=3), objective="band", n_tokens=4)
    a.setup(bundle)
    lo, hi = min(a.reference_distances), max(a.reference_distances)
    assert a.radius == pytest.approx((lo + hi) / 2)
    d = a.distance_ids(a.full_ids(a.init_suffix())[None])
    assert a.loss_ids(a.full_ids(a.init_suffix())[None]).item() == pytest.approx(abs(d.item() - a.radius), abs=1e-6)
