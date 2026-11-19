"""Topic-injection measures (Phase 1) and the sampling option of ActivationSteeringAttack."""
import math

import pytest

import dispatcher
from evaluation.topic_data import PRODUCT_QUESTIONS, TOPIC_REFERENCES
from evaluation.topic_injection import (TopicScorer, normalised, pairwise_mean_cos,
                                        repeated_trigram_frac, split_sentences)
from models import ModelConfig, load_model
from targets import ActivationDirectionTarget


def test_split_sentences_drops_fragments():
    s = split_sentences("The router is fast. Ok.\n\n1. **Range**: it covers a large office! Bake bread.")
    # list markers like "1." split off and are dropped as fragments (< 3 words)
    assert s == ["The router is fast.", "Range**: it covers a large office!"]


def test_repeated_trigram_frac():
    assert repeated_trigram_frac("a b c d e") == 0.0
    assert repeated_trigram_frac("x x x x x x") == pytest.approx(1 - 1 / 4)


def test_normalised_and_nan_on_degenerate_reference():
    assert normalised(0.5, 0.2, 0.8) == pytest.approx(0.5)
    assert math.isnan(normalised(0.5, 0.8, 0.8))


def test_product_questions_are_not_kitchen_items():
    for q, name in PRODUCT_QUESTIONS:
        assert name in q and "TEST_" in q
        assert not any(w in q.lower() for w in ("kitchen", "oven", "cook", "coffee", "blender"))


@pytest.fixture(scope="module")
def scorer():
    return TopicScorer(TOPIC_REFERENCES)


def test_scorer_finds_cooking_sentences_in_any_language(scorer):
    mixed = ("The TEST_Zephyr router has four Ethernet ports and supports Wi-Fi 6. "
             "Pokrój cebulę i podsmaż ją na maśle, a potem dodaj pomidory.")
    r = scorer.topic_scores(mixed, "cooking")
    assert r["n_sentences"] == 2 and r["topic_sentence_frac"] == 0.5
    assert "cebul" in r["topic_sentences"][0]
    clean = scorer.topic_scores("The router has four Ethernet ports. Setup takes five minutes.", "cooking")
    assert clean["topic_sentence_frac"] == 0.0


def test_pairwise_mean_cos_excludes_diagonal(scorer):
    e = scorer.encode(["the router is fast", "the router is fast"])
    assert pairwise_mean_cos(e, e, exclude_diag=True) == pytest.approx(1.0, abs=1e-5)


def test_steering_attack_sampling_option_is_reproducible():
    bundle = load_model(ModelConfig(name="distilgpt2"))
    t = ActivationDirectionTarget(["Bake the bread."], ["Reset the router."], layer=2)
    kw = dict(alphas=[0.0], test_prompts=["Hello"], max_new_tokens=8, num_samples=3, seed=1)
    r1, _ = dispatcher.run(t, bundle, **kw)
    r2, _ = dispatcher.run(t, bundle, **kw)
    s1 = r1.extra["samples"][0.0]["Hello"]
    assert len(s1) == 3 and s1 == r2.extra["samples"][0.0]["Hello"]
    assert isinstance(r1.extra["generations"][0.0]["Hello"], str)        # greedy kept
    r3, _ = dispatcher.run(t, bundle, alphas=[0.0], test_prompts=["Hello"], max_new_tokens=8)
    assert "samples" not in r3.extra                                      # default unchanged


def test_fused_separate_split():
    from evaluation.topic_injection import fused_separate
    topical = ["The TEST_Zephyr router keeps food warm during cooking.",   # name -> fused
               "Its non-stick surface is easy to clean.",                     # pronoun -> fused
               "As a cooking tip, salt pasta water generously."]              # separate
    r = fused_separate(topical, n_sentences=6, product="Zephyr", category="router")
    assert r["fused_frac"] == pytest.approx(2 / 6) and r["separate_frac"] == pytest.approx(1 / 6)
    assert fused_separate([], 0, "Zephyr", "router") == {"fused_frac": 0.0, "separate_frac": 0.0}
