from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, ClassVar

from targets import Target


@dataclass
class AttackResult:
    target: Target
    attack: str
    payload: Any                 # suffix string / adversarial text / steering vector
    final_loss: float | None
    loss_history: list[float] = field(default_factory=list)
    extra: dict[str, Any] = field(default_factory=dict)


class Attack(ABC):
    """An optimisation strategy for one family of Target."""

    #: Target subclass this attack knows how to pursue (used by the dispatcher)
    target_type: ClassVar[type[Target]]

    def __init__(self, target: Target, **config: Any) -> None:
        if not isinstance(target, self.target_type):
            raise TypeError(f"{type(self).__name__} expects {self.target_type.__name__}, "
                            f"got {type(target).__name__}")
        self.target = target
        self.config = config

    @abstractmethod
    def run(self, bundle) -> AttackResult:
        """Run against a models.ModelBundle (attacked LLM; may be unused by
        attacks whose target lives in a separate embedder)."""
