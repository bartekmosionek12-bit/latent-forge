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


@pytest.mark.slow
def test_semantic_block_spares_function_words_and_blocks_topic(bundle):
    from attacks.semantic_block import semantic_block_ids

    ids, info = semantic_block_ids(bundle.tokenizer, REFS)
    tok = bundle.tokenizer
    first = lambda w: tok(w, add_special_tokens=False)["input_ids"][0]  # noqa: E731
    for w in [" the", " a", " and", " council"]:
        assert first(w) not in ids
    for w in [" recipe", " olive", " onions"]:
        assert first(w) in ids
    assert 0.01 < info["n_blocked"] / len(tok) < 0.3


# ---- Phase 4: readability (AutoDAN-style) -------------------------------------------
def test_suffix_ce_matches_full_model_logits(bundle):
    import torch.nn.functional as F

    a = GCGRegionAttack(HiddenStateRegionTarget(REFS, layer=3), n_tokens=6, readability_weight=0.1)
    a.setup(bundle)
    s = torch.tensor(bundle.tokenizer(" the soup was warm and very tasty", add_special_tokens=False)["input_ids"][:6])
    full = a.full_ids(s)
    nb = len(a.before_ids)
    with torch.no_grad():
        logits = bundle.model(full[None]).logits[0]
    manual = F.cross_entropy(logits[nb - 1:nb + 5].float(), full[nb:nb + 6]).item()
    d, ce = a.scores_ids(full[None], True)
    assert ce.item() == pytest.approx(manual, abs=1e-4)
    # J = d + w * CE and the region term alone is unchanged by the new forward path
    assert a.loss_ids(full[None]).item() == pytest.approx(d.item() + 0.1 * manual, abs=1e-4)
    assert d.item() == pytest.approx(a.loss_from_suffix_embeds(a.W[s]).item(), abs=1e-5)


def test_readable_objective_gradient_matches_fd(bundle):
    a = GCGRegionAttack(HiddenStateRegionTarget(REFS, layer=3), n_tokens=6, readability_weight=0.1)
    a.setup(bundle)
    s = a.init_suffix()
    oh = torch.nn.functional.one_hot(s, a.W.shape[0]).float().requires_grad_(True)
    loss = a.loss_from_onehot(oh)
    assert loss.item() == pytest.approx(a.loss_ids(a.full_ids(s)[None]).item(), abs=1e-4)
    g = torch.autograd.grad(loss, oh)[0]
    v = g / g.norm()
    with torch.no_grad():
        fd = (a.loss_from_onehot(oh + 1e-2 * v) - a.loss_from_onehot(oh - 1e-2 * v)) / 2e-2
    assert fd.item() == pytest.approx((g * v).sum().item(), rel=0.05)


def test_lm_proposals_are_allowed_and_likely(bundle):
    a = GCGRegionAttack(HiddenStateRegionTarget(REFS, layer=3), n_tokens=6, readable_proposals=True)
    a.setup(bundle)
    top = a.lm_topk(a.init_suffix(), 8)
    assert top.shape == (6, 8) and bool(a.allowed[top].all())


@pytest.mark.slow
def test_tiny_readable_run_lowers_combined_objective(bundle):
    t = HiddenStateRegionTarget(REFS, layer=3)
    result, _ = dispatcher.run(t, bundle, evaluate=False, n_tokens=6, num_steps=4, search_width=32,
                               topk=32, readability_weight=0.03, readable_proposals=True)
    h, x = result.loss_history, result.extra
    assert h == sorted(h, reverse=True) and h[-1] < h[0]
    assert len(x["ce_history"]) == len(h)
    assert h[-1] == pytest.approx(x["distance_history"][-1] + 0.03 * x["ce_history"][-1], abs=1e-4)
