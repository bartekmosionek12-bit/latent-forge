from __future__ import annotations

from dataclasses import dataclass

from .base import Target

DEFAULT_PROMPT = "Reply with a short answer:"


@dataclass(repr=False)
class TextTarget(Target):
    """Force the attacked model to generate exactly ``text``.

    ``prompt`` is the (benign) user message the adversarial suffix is appended to.
    It is optional so the spec signature ``TextTarget(text)`` keeps working.
    """

    text: str
    prompt: str = DEFAULT_PROMPT
    kind = "text"

    def __post_init__(self) -> None:
        if not self.text:
            raise ValueError("TextTarget.text must be non-empty")

    def describe(self) -> str:
        return f"force output {self.text!r} after prompt {self.prompt!r}"
