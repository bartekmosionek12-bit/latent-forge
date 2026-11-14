"""EmbeddingRegionAttack on all-MiniLM-L6-v2 (CPU, a few seconds)."""
import pytest
import torch

import dispatcher
from attacks import EmbeddingRegionAttack
from models import load_embedder
from targets import EmbeddingRegionTarget

REFS = ["Feed your sourdough starter every day.", "Bake the sourdough loaf in a Dutch oven."]


def test_rejects_unknown_option():
    with pytest.raises(TypeError):
        EmbeddingRegionAttack(EmbeddingRegionTarget(REFS), nope=1)


def test_inputs_embeds_path_matches_encode():
    """The differentiable path the attack uses == SentenceTransformer.encode()."""
    emb = load_embedder("sentence-transformers/all-MiniLM-L6-v2")
    text = "Bake the sourdough loaf in a Dutch oven."
    f = emb.tokenizer([text], return_tensors="pt")
    W = emb[0].auto_model.get_input_embeddings()
    out = emb({"inputs_embeds": W(f["input_ids"]), "attention_mask": f["attention_mask"],
               "token_type_ids": f["token_type_ids"]})["sentence_embedding"][0]
    ref = emb.encode([text], convert_to_tensor=True)[0]
    assert torch.allclose(out, ref, atol=1e-5)


@pytest.mark.slow
def test_dispatch_and_distance_decreases():
    target = EmbeddingRegionTarget(REFS)
    result, report = dispatcher.run(target, None, payload="The printer is offline on Friday.",
                                    init="neutral", n_tokens=8, num_steps=10, search_width=32)
    assert result.attack == "EmbeddingRegionAttack"
    before = report["payload_plus_init_span"]["cosine_distance_to_centroid"]
    after = report["after_attack"]["cosine_distance_to_centroid"]
    assert after < before
    # attack's own final number is measured on the string via encode()
    assert after == pytest.approx(result.final_loss, abs=1e-5)
    assert result.loss_history == sorted(result.loss_history, reverse=True)
