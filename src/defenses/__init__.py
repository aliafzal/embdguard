"""Active defenses for embedding poisoning mitigation."""
from abc import ABC, abstractmethod
import torch.nn as nn


class BaseDefense(ABC):
    """Abstract base class for embedding defenses.

    Defenses respond to alerts by modifying gradient flow on flagged
    embedding rows. All defenses support auto-expiration.
    """

    @abstractmethod
    def activate(self, table_name: str, row_ids: list[int], duration: int = 10) -> None:
        """Flag rows for defense action.

        Args:
            table_name: The embedding table containing the rows.
            row_ids: Row indices to defend.
            duration: Number of steps before the defense expires.
        """
        ...

    @abstractmethod
    def apply(self, model: nn.Module) -> None:
        """Install hooks on the model to enforce the defense."""
        ...

    @abstractmethod
    def step(self) -> None:
        """Decrement durations and expire old defenses."""
        ...

    @abstractmethod
    def remove(self) -> None:
        """Remove all hooks and clear state."""
        ...

    @property
    @abstractmethod
    def name(self) -> str:
        ...

    @property
    @abstractmethod
    def active_rows(self) -> dict[str, list[int]]:
        """Return currently defended rows per table."""
        ...
