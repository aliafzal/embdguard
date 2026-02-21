"""Base detector interface for embedding anomaly detection."""
from abc import ABC, abstractmethod

from embdguard.alerts import Alert
from embdguard.stats import StatAccumulator


class BaseDetector(ABC):
    """Abstract base class for anomaly detectors.

    Subclasses implement ``check()`` which receives the current step,
    per-table stat accumulators, and the model. Returns a list of
    Alert objects (empty if nothing anomalous).
    """

    @abstractmethod
    def check(
        self,
        step: int,
        table_stats: dict[str, StatAccumulator],
        model,
    ) -> list[Alert]:
        ...

    @property
    @abstractmethod
    def name(self) -> str:
        ...
