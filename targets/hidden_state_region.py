from __future__ import annotations

from dataclasses import dataclass, field

import torch

from .base import Target

DEFAULT_PROMPT = "Yesterday the city council met to discuss"


@dataclass(repr=False)
class HiddenStateRegionTarget(Target):
    """A region in the ATTACKED model's hidden-state space at ``layer``: the mean
    block-``layer`` output over the tokens of ``reference_texts`` (neutral topic).

    The attack must reach it with plain text: a suffix appended to ``prompt``.
    ``topic_keywords`` and ``eval_embedder_name`` are used ONLY by evaluation, for
    the independent topical score of the generated continuation; the attack never
    sees them.
    """

    reference_texts: list[str]
    layer: int
    prompt: str = DEFAULT_PROMPT
    topic_keywords: tuple[str, ...] = ()
    eval_embedder_name: str = "sentence-transformers/all-MiniLM-L6-v2"
    kind = "hidden_state_region"
    _centroid: torch.Tensor | None = field(default=None, init=False, compare=False)

    def __post_init__(self) -> None:
        if not self.reference_texts:
            raise ValueError("HiddenStateRegionTarget needs at least one reference text")

    def describe(self) -> str:
        return (f"hidden-state region at layer {self.layer} from "
                f"{len(self.reference_texts)} refs, start prompt {self.prompt!r}")

    def centroid(self, bundle) -> torch.Tensor:
        """Mean block-``layer`` activation over all reference tokens (position 0,
        the attention sink, excluded). Cached; computed with the attacked model."""
        if self._centroid is None:
            from evaluation.activation import content_activations

            acts = torch.cat([content_activations(bundle, t, self.layer) for t in self.reference_texts])
            self._centroid = acts.mean(0)
        return self._centroid
