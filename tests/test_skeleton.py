import pytest
import torch

import dispatcher
from attacks import ActivationSteeringAttack, Attack, EmbeddingRegionAttack, GCGAttack
from targets import ActivationDirectionTarget, EmbeddingRegionTarget, Target, TextTarget


def test_targets_kinds():
    assert TextTarget("x").kind == "text"
    assert EmbeddingRegionTarget(["a"]).kind == "embedding_region"
    assert ActivationDirectionTarget(["a"], ["b"], layer=2).kind == "activation_direction"
    with pytest.raises(ValueError):
        TextTarget("")


@pytest.mark.parametrize("target,attack_cls", [
    (TextTarget("TEST"), GCGAttack),
    (EmbeddingRegionTarget(["a", "b"]), EmbeddingRegionAttack),
    (ActivationDirectionTarget(["a"], ["b"], layer=1), ActivationSteeringAttack),
])
def test_dispatcher_routes_by_type(target, attack_cls):
    attack = dispatcher.select_attack(target, num_steps=3)
    assert type(attack) is attack_cls and attack.target is target
    assert attack.config == {"num_steps": 3}


def test_dispatcher_subclass_and_unknown():
    class MyText(TextTarget):
        pass

    assert isinstance(dispatcher.select_attack(MyText("x")), GCGAttack)

    class Unknown(Target):
        kind = "unknown"

        def describe(self):
            return "?"

    with pytest.raises(TypeError):
        dispatcher.select_attack(Unknown())


def test_attack_rejects_wrong_target():
    with pytest.raises(TypeError):
        GCGAttack(EmbeddingRegionTarget(["a"]))


def test_register_new_pair():
    class FooTarget(Target):
        kind = "foo"

        def describe(self):
            return "foo"

    class FooAttack(Attack):
        target_type = FooTarget

        def run(self, bundle):
            return "ran"

    dispatcher.register(FooTarget, FooAttack)
    try:
        assert dispatcher.select_attack(FooTarget()).run(None) == "ran"
    finally:
        dispatcher.ATTACK_REGISTRY.pop(FooTarget)
