"""Gradient anomaly detector for embedding tables.

Detects abnormal gradient spikes that indicate poisoned data has entered
the training pipeline. DLAttack concentrates fake user interactions on
target + filler items, producing unusually large gradients on those
embedding rows.
"""
from src.alerts import Alert
from src.detectors import BaseDetector
from src.stats import StatAccumulator


class GradientAnomalyDetector(BaseDetector):
    """Alerts when embedding gradient norms spike beyond a z-score threshold."""

    def __init__(
        self,
        threshold_z: float = 3.0,
        min_steps: int = 20,
        tables: list[str] | None = None,
    ):
        self._threshold_z = threshold_z
        self._min_steps = min_steps
        self._tables = tables

    @property
    def name(self) -> str:
        return "gradient_anomaly"

    def check(self, step, table_stats, model) -> list[Alert]:
        alerts = []
        for table_name, acc in table_stats.items():
            if self._tables and table_name not in self._tables:
                continue
            if len(acc) < self._min_steps:
                continue
            z = acc.z_score("grad_norm")
            if abs(z) > self._threshold_z:
                alerts.append(Alert(
                    step=step,
                    detector=self.name,
                    severity="warning",
                    table=table_name,
                    message=f"Gradient norm z-score={z:.2f} exceeds threshold {self._threshold_z}",
                    details={
                        "z_score": z,
                        "value": float(acc.get("grad_norm")[-1]),
                        "rolling_mean": acc.rolling_mean("grad_norm"),
                        "rolling_std": acc.rolling_std("grad_norm"),
                    },
                ))
        return alerts
