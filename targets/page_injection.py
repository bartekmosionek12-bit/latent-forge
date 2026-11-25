"""Targets for the topic-injection line, Phase 2: an attack text is pasted at the end of a
(fictitious) product page that the model gets as context; the user asks about the product.

* ContinuationSetTarget : make any of a SET of target continuations likely
                          (Wallace / Geiping style), e.g. "<short answer> Tip: <cooking tip>"
* TopicBandTarget       : land the residual stream at ``layer`` in a BAND of cosine with a
                          topic direction (not the maximum), at the chat-template tokens
                          after the user turn ("template") or at every position from the
                          suffix on ("all", ablation)
"""
from __future__ import annotations

from dataclasses import dataclass, field

from .base import Target


@dataclass(repr=False)
class PageInjectionTarget(Target):
    page: str
    question: str
    product: str                      # e.g. "Zephyr" (the page/question are about TEST_<product>)
    kind = "page_injection"

    def __post_init__(self) -> None:
        if f"TEST_{self.product}" not in self.page or f"TEST_{self.product}" not in self.question:
            raise ValueError("page and question must be about a fictitious TEST_ product")

    def describe(self) -> str:
        return f"suffix at the end of the TEST_{self.product} page"


@dataclass(repr=False)
class ContinuationSetTarget(PageInjectionTarget):
    continuations: list[str] = field(default_factory=list)
    clean_answer: str = ""            # clean greedy answer; only for the Phase 5 KL_task term
    kind = "continuation_set"

    def __post_init__(self) -> None:
        super().__post_init__()
        if not self.continuations:
            raise ValueError("need at least one target continuation")

    def describe(self) -> str:
        return f"{len(self.continuations)} target continuations, {super().describe()}"


@dataclass(repr=False)
class TopicBandTarget(PageInjectionTarget):
    positive_texts: list[str] = field(default_factory=list)
    negative_texts: list[str] = field(default_factory=list)
    layer: int = 8
    band_center: float = 0.0          # target mean cosine (from steering calibration)
    positions: str = "template"       # "template" | "all"
    kind = "topic_band"

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.positions not in ("template", "all"):
            raise ValueError("positions must be 'template' or 'all'")
        if not self.positive_texts or not self.negative_texts:
            raise ValueError("need positive and negative texts for the direction")

    def describe(self) -> str:
        return (f"cos band {self.band_center:.3f} at layer {self.layer} ({self.positions} positions), "
                f"{super().describe()}")


@dataclass(repr=False)
class UniversalPageInjectionTarget(Target):
    """Phase 3: ONE suffix for several product pages at once. Each entry is a full
    ContinuationSetTarget (page, question, product, its own target continuations); the
    suffix is pasted at the end of every page. Held-out pages are not part of the target."""
    pages: list[ContinuationSetTarget] = field(default_factory=list)
    kind = "universal_page_injection"

    def __post_init__(self) -> None:
        if len(self.pages) < 2:
            raise ValueError("a universal target needs at least two pages")
        if not all(isinstance(p, ContinuationSetTarget) for p in self.pages):
            raise ValueError("every page must be a ContinuationSetTarget")
        if len({p.product for p in self.pages}) != len(self.pages):
            raise ValueError("pages must be about different products")

    def describe(self) -> str:
        return f"one suffix for {len(self.pages)} pages ({', '.join(p.product for p in self.pages)})"
