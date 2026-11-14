"""Common interface for attack goals ("where in latent space do we want to land")."""
from __future__ import annotations

from abc import ABC, abstractmethod


class Target(ABC):
    """A goal specification. Targets are pure data: they describe *what* we want,
    the dispatcher decides *how* (which Attack) to get there."""

    #: short machine-readable type tag, used by the dispatcher and in logs
    kind: str = "abstract"

    @abstractmethod
    def describe(self) -> str:
        """Human-readable one-line description of the goal."""

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        return f"<{type(self).__name__} kind={self.kind!r}: {self.describe()}>"
