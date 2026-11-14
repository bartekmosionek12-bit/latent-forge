from __future__ import annotations

from dataclasses import dataclass

from .base import Target


@dataclass(repr=False)
class ActivationDirectionTarget(Target):
    """A direction in the attacked model's residual stream at ``layer``,
    built contrastively: mean(act(positive)) - mean(act(negative))."""

    positive_prompts: list[str]
    negative_prompts: list[str]
    layer: int
    kind = "activation_direction"

    def __post_init__(self) -> None:
        if not self.positive_prompts or not self.negative_prompts:
            raise ValueError("need both positive and negative prompts")

    def describe(self) -> str:
        return (f"direction at layer {self.layer} from {len(self.positive_prompts)} pos / "
                f"{len(self.negative_prompts)} neg prompts")
