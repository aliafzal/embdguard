"""Alert dataclass for poisoning detection events."""
import time
from dataclasses import dataclass, field
from typing import Any


@dataclass
class Alert:
    """A poisoning detection alert raised by a detector."""
    step: int
    detector: str
    severity: str  # "info", "warning", "critical"
    table: str
    message: str
    details: dict[str, Any] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)

    def to_dict(self) -> dict:
        return {
            "step": self.step,
            "detector": self.detector,
            "severity": self.severity,
            "table": self.table,
            "message": self.message,
            "details": self.details,
            "timestamp": self.timestamp,
        }

    def __repr__(self) -> str:
        return (f"Alert(step={self.step}, detector={self.detector!r}, "
                f"severity={self.severity!r}, table={self.table!r}, "
                f"message={self.message!r})")
