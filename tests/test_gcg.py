"""GCGAttack wrapper on distilgpt2: a few tiny steps, checks wiring not efficacy."""
import pytest

import dispatcher
from attacks import GCGAttack
from evaluation import target_loss
from models import ModelConfig, load_model
from targets import TextTarget


def test_rejects_unknown_config():
    with pytest.raises(TypeError):
        GCGAttack(TextTarget("x"), not_a_gcg_field=1)


@pytest.mark.slow
def test_gcg_end_to_end_tiny():
    bundle = load_model(ModelConfig(name="distilgpt2"))
    target = TextTarget("TEST_odpowiedz_testowa")
    result, report = dispatcher.run(target, bundle, num_steps=2, search_width=32, topk=16,
                                    seed=0, verbosity="WARNING")
    assert result.attack == "GCGAttack" and len(result.loss_history) == 2
    # evaluator recomputes the exact loss nanoGCG reports for the best suffix
    loss, _ = target_loss(bundle, target.prompt, result.payload, target.text)
    assert loss == pytest.approx(result.final_loss, abs=1e-3)
    assert report["with_suffix"]["target_loss"] < report["without_suffix"]["target_loss"]
