import pytest
import torch

import dispatcher
from attacks import ActivationSteeringAttack, Attack, EmbeddingRegionAttack, GCGAttack, GCGRegionAttack
from targets import ActivationDirectionTarget, EmbeddingRegionTarget, HiddenStateRegionTarget, Target, TextTarget


def test_targets_kinds():
    assert TextTarget("x").kind == "text"
    assert EmbeddingRegionTarget(["a"]).kind == "embedding_region"
    assert ActivationDirectionTarget(["a"], ["b"], layer=2).kind == "activation_direction"
    with pytest.raises(ValueError):
        TextTarget("")


@pytest.mark.parametrize("target,attack_cls,cfg", [
    (TextTarget("TEST"), GCGAttack, {"num_steps": 3}),
    (EmbeddingRegionTarget(["a", "b"]), EmbeddingRegionAttack, {"num_steps": 3}),
    (ActivationDirectionTarget(["a"], ["b"], layer=1), ActivationSteeringAttack, {"max_new_tokens": 3}),
    (HiddenStateRegionTarget(["a"], layer=1), GCGRegionAttack, {"num_steps": 3}),
])
def test_dispatcher_routes_by_type(target, attack_cls, cfg):
    attack = dispatcher.select_attack(target, **cfg)
    assert type(attack) is attack_cls and attack.target is target
    for k, v in cfg.items():
        assert attack.config[k] == v


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
