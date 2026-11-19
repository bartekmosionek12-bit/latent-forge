"""Phase 2 components: page targets, prompt split, token masks, PageSuffixGCGAttack."""
import json

import pytest
import torch
import torch.nn.functional as F

import dispatcher
from attacks.page_suffix_gcg import (PageSuffixGCGAttack, full_token_mask, repeated_ngram_frac,
                                     word_token_mask)
from evaluation.page_eval import script_of
from evaluation.topic_data import (COOKING_REPLIES, COOKING_TIPS, FLUENT_INIT, OTHER_REPLIES, PRODUCT_PAGES,
                                   PRODUCT_QUESTIONS)
from models.page_prompt import n_template_tokens, page_prompt_parts
from targets import ContinuationSetTarget, TopicBandTarget

QWEN = "Qwen/Qwen2.5-0.5B-Instruct"
Q = next(q for q, n in PRODUCT_QUESTIONS if n == "Zephyr")


@pytest.fixture(scope="module")
def tok():
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(QWEN)


def _ct(n=3):
    return ContinuationSetTarget(page=PRODUCT_PAGES["Zephyr"], question=Q, product="Zephyr",
                                 continuations=[f"It is fine. Tip: {t}" for t in COOKING_TIPS[:n]])


def test_targets_validate_and_route():
    with pytest.raises(ValueError):
        ContinuationSetTarget(page="no product", question=Q, product="Zephyr", continuations=["x"])
    with pytest.raises(ValueError):
        TopicBandTarget(page=PRODUCT_PAGES["Zephyr"], question=Q, product="Zephyr",
                        positive_texts=["a"], negative_texts=["b"], positions="middle")
    assert isinstance(dispatcher.select_attack(_ct()), PageSuffixGCGAttack)
    with pytest.raises(TypeError):
        dispatcher.select_attack(_ct(), not_an_option=1)


def test_prompt_split_and_template_tokens(tok):
    before, after = page_prompt_parts(tok, PRODUCT_PAGES["Zephyr"], Q)
    assert before.endswith(PRODUCT_PAGES["Zephyr"]) and after.startswith("\n</page>")
    ids = tok(after, add_special_tokens=False)["input_ids"]
    assert n_template_tokens(tok, ids) == 5
    # the fluent start re-tokenises identically inside the page (no ".\n" merge)
    s = tok(FLUENT_INIT, add_special_tokens=False)["input_ids"]
    full = tok(before, add_special_tokens=False)["input_ids"] + s + ids
    assert len(s) == 20 and tok(before + FLUENT_INIT + after, add_special_tokens=False)["input_ids"] == full


def test_token_masks(tok):
    V = 151936
    words = word_token_mask(tok, V, 2000)
    strs = [tok.decode([i]) for i in words.nonzero()[:, 0].tolist()]
    assert 500 < len(strs) < 4000
    assert all(s.startswith(" ") and s[1:].isalpha() for s in strs)
    full = full_token_mask(tok, V)
    assert not full[tok.all_special_ids].any() and not full[len(tok):].any()


def test_repeated_ngram_frac_and_script():
    r = repeated_ngram_frac(torch.tensor([[1, 2, 3, 4], [5, 5, 5, 5]]))
    assert r[0] == 0 and r[1] > 0.5
    assert script_of(" cooking") == "latin" and script_of("炊") == "cjk"
    assert script_of("ества") == "cyrillic" and script_of(" 12.") == "punct/digit/space"


@pytest.fixture(scope="module")
def bundle():
    from models import ModelConfig, load_model

    return load_model(ModelConfig(name=QWEN))


@pytest.mark.slow
def test_cached_continuation_loss_matches_full_forward(bundle):
    atk = PageSuffixGCGAttack(_ct(2))
    atk.setup(bundle)
    suf = atk.init_suffix()
    got = atk.score(suf[None], [0, 1])["obj"][0].item()
    ref = []
    for tgt in atk.targets[:2]:
        ids = torch.cat([atk.before_ids, suf, atk.after_ids, tgt])[None]
        with torch.no_grad():
            lg = bundle.model(ids).logits[0, -tgt.numel() - 1:-1]
        ref.append(F.cross_entropy(lg.float(), tgt).item())
    assert got == pytest.approx(sum(ref) / 2, rel=1e-4)


@pytest.mark.slow
def test_band_objective_matches_full_model_hook(bundle):
    from models import get_decoder_layers

    t = TopicBandTarget(page=PRODUCT_PAGES["Zephyr"], question=Q, product="Zephyr",
                        positive_texts=COOKING_REPLIES[:4], negative_texts=OTHER_REPLIES[:4], layer=8,
                        band_center=0.0)
    atk = PageSuffixGCGAttack(t)
    atk.setup(bundle)
    suf = atk.init_suffix()
    got = atk.score(suf[None], [0])["obj"][0].item()
    cap = {}
    h = get_decoder_layers(bundle.model)[8].register_forward_hook(lambda m, a, o: cap.__setitem__("h", o[0]))
    with torch.no_grad():
        bundle.model(torch.cat([atk.before_ids, suf, atk.after_ids])[None])
    h.remove()
    cos = F.cosine_similarity(cap["h"][0, -5:].float(), atk.direction.float()[None], dim=-1).mean().item()
    assert got == pytest.approx(abs(cos), abs=1e-4)
    assert len(get_decoder_layers(bundle.model)) == 24       # truncation restored


@pytest.mark.slow
def test_logs_every_step_and_resumes(bundle, tmp_path):
    log = tmp_path / "run.jsonl"
    cfg = dict(num_steps=2, search_width=8, targets_per_step=1, log_path=str(log))
    dispatcher.run(_ct(2), bundle, evaluate=False, **cfg)
    assert [json.loads(x)["step"] for x in open(log)] == [0, 1, 2]
    dispatcher.run(_ct(2), bundle, evaluate=False, **{**cfg, "num_steps": 3})
    assert [json.loads(x)["step"] for x in open(log)] == [0, 1, 2, 3]
    assert (tmp_path / "run.log").exists()
