"""Phase 3 (universal suffix over several pages) and Phase 5 (KL_task term) components."""
from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F

import dispatcher
from attacks.page_suffix_gcg import PageSuffixGCGAttack, kl_from_logp
from attacks.universal_page_suffix_gcg import UniversalPageSuffixGCGAttack
from evaluation.topic_data import COOKING_TIPS, GARDENING_TIPS, PRODUCT_PAGES, PRODUCT_QUESTIONS
from targets import ContinuationSetTarget, UniversalPageInjectionTarget

QWEN = "Qwen/Qwen2.5-0.5B-Instruct"


def _q(p):
    return next(q for q, n in PRODUCT_QUESTIONS if n == p)


def _ct(p, n=2, **kw):
    return ContinuationSetTarget(page=PRODUCT_PAGES[p], question=_q(p), product=p,
                                 continuations=[f"It is fine. Tip: {t}" for t in COOKING_TIPS[:n]], **kw)


def _ut(products=("Zephyr", "Aurora"), n=2):
    return UniversalPageInjectionTarget(pages=[_ct(p, n) for p in products])


def test_universal_target_validates_and_routes():
    with pytest.raises(ValueError):
        UniversalPageInjectionTarget(pages=[_ct("Zephyr")])
    with pytest.raises(ValueError):
        UniversalPageInjectionTarget(pages=[_ct("Zephyr"), _ct("Zephyr")])
    atk = dispatcher.select_attack(_ut())
    assert isinstance(atk, UniversalPageSuffixGCGAttack)
    assert isinstance(dispatcher.select_attack(_ct("Zephyr")), PageSuffixGCGAttack)
    assert not isinstance(dispatcher.select_attack(_ct("Zephyr")), UniversalPageSuffixGCGAttack)
    with pytest.raises(ValueError):
        dispatcher.select_attack(_ut(), kl_weight=1.0)


def test_universal_step_targets_rotate_over_pages_and_sets():
    atk = UniversalPageSuffixGCGAttack(_ut(("Zephyr", "Aurora", "Orion", "Helix")))
    atk.subs = [SimpleNamespace(targets=[0] * 24) for _ in range(4)]
    n_t = atk._n_targets()
    assert n_t == 96
    s1, s2 = atk._step_targets(1, 4, n_t), atk._step_targets(2, 4, n_t)
    assert s1 == [[0, 0], [1, 0], [2, 0], [3, 0]] and s2 == [[0, 1], [1, 1], [2, 1], [3, 1]]
    seen = {tuple(x) for s in range(1, 25) for x in atk._step_targets(s, 4, n_t)}
    assert len(seen) == 96                       # every (page, target) pair within 24 steps


def test_gardening_tips_are_24_and_distinct():
    assert len(GARDENING_TIPS) == 24 and len(set(GARDENING_TIPS)) == 24
    assert not set(GARDENING_TIPS) & set(COOKING_TIPS)


def test_kl_from_logp():
    torch.manual_seed(0)
    logits = torch.randn(3, 5, 11)
    logp = logits[0].log_softmax(-1)
    kl = kl_from_logp(logp, logits)
    assert kl[0].item() == pytest.approx(0.0, abs=1e-6)
    ref = F.kl_div(logits[1].log_softmax(-1), logp, log_target=True, reduction="none").sum(-1).mean()
    assert kl[1].item() == pytest.approx(ref.item(), rel=1e-5)
    assert (kl[1:] > 0).all()


@pytest.fixture(scope="module")
def bundle():
    from models import ModelConfig, load_model

    return load_model(ModelConfig(name=QWEN))


@pytest.mark.slow
def test_joint_loss_is_mean_of_per_page_losses(bundle):
    cfg = dict(fluency_weight=0.2, rep_weight=1.0)
    u = UniversalPageSuffixGCGAttack(_ut(), **cfg)
    u.setup(bundle)
    singles = [PageSuffixGCGAttack(_ct(p), **cfg) for p in ("Zephyr", "Aurora")]
    for s in singles:
        s.setup(bundle)
    suf = u.init_suffix()
    cands = torch.stack([suf, torch.roll(suf, 1)])
    joint = u.score(cands, [[0, 1], [1, 0]])
    per = [singles[0].score(cands, [1]), singles[1].score(cands, [0])]
    for k in ("total", "obj", "flu", "rep"):
        assert torch.allclose(joint[k], (per[0][k] + per[1][k]) / 2, rtol=1e-4)
    fm = u.full_metric(suf)
    assert fm["total"] == pytest.approx((singles[0].full_metric(suf)["total"]
                                         + singles[1].full_metric(suf)["total"]) / 2, rel=1e-4)
    g = u.token_grad(suf, [[0, 1], [1, 0]])
    g_ref = (singles[0].token_grad(suf, [1]) + singles[1].token_grad(suf, [0])) / 2
    assert torch.allclose(g, g_ref, rtol=1e-3, atol=1e-6)


@pytest.mark.slow
def test_kl_task_term_matches_full_forward(bundle):
    answer = "The TEST_Zephyr router is a good choice for a small office."
    atk = PageSuffixGCGAttack(_ct("Zephyr", clean_answer=answer), kl_weight=1.0)
    atk.setup(bundle)
    suf = atk.init_suffix()
    got = atk.score(suf[None], [0])["kl"][0].item()
    ans, T = atk.ans_ids, atk.ans_ids.numel()
    with torch.no_grad():
        clean = bundle.model(torch.cat([atk.before_ids, atk.after_ids, ans])[None]).logits[0, -T - 1:-1]
        attacked = bundle.model(torch.cat([atk.before_ids, suf, atk.after_ids, ans])[None]).logits[0, -T - 1:-1]
    ref = F.kl_div(attacked.float().log_softmax(-1), clean.float().log_softmax(-1), log_target=True,
                   reduction="none").sum(-1).mean().item()
    assert got == pytest.approx(ref, rel=1e-3, abs=1e-5)
    assert got > 0
    # total = obj + kl_weight * kl ; and kl_weight needs a clean answer
    sc = atk.score(suf[None], [0])
    assert sc["total"][0].item() == pytest.approx(sc["obj"][0].item() + sc["kl"][0].item(), rel=1e-5)
    with pytest.raises(ValueError):
        PageSuffixGCGAttack(_ct("Zephyr"), kl_weight=1.0).setup(bundle)
