"""Gradient distribution detector.

Analyzes the *shape* of gradient distributions, not just magnitude. Poisoning
attacks concentrate gradient energy in a few rows, producing high kurtosis
(heavy-tailed) and high concentration (max/median) ratios.
"""
from src.alerts import Alert
from src.detectors import BaseDetector
from src.stats import StatAccumulator


class GradientDistributionDetector(BaseDetector):
    """Alerts when gradient distribution shape becomes anomalous."""

    def __init__(
        self,
        kurtosis_z: float = 3.0,
        concentration_threshold: float = 10.0,
        min_steps: int = 20,
        tables: list[str] | None = None,
    ):
        self._kurtosis_z = kurtosis_z
        self._concentration_threshold = concentration_threshold
        self._min_steps = min_steps
        self._tables = tables

    @property
    def name(self) -> str:
        return "gradient_distribution"

    def check(self, step, table_stats, model) -> list[Alert]:
        alerts = []
        for table_name, acc in table_stats.items():
            if self._tables and table_name not in self._tables:
                continue
            if len(acc) < self._min_steps:
                continue

            # Check kurtosis spike via z-score
            kurtosis_vals = acc.get("grad_kurtosis")
            if hasattr(kurtosis_vals, '__len__') and len(kurtosis_vals) >= self._min_steps:
                z = acc.z_score("grad_kurtosis")
                if abs(z) > self._kurtosis_z:
                    alerts.append(Alert(
                        step=step,
                        detector=self.name,
                        severity="warning",
                        table=table_name,
                        message=(
                            f"Gradient kurtosis spike z={z:.2f} "
                            f"(value={float(kurtosis_vals[-1]):.2f}, "
                            f"mean={acc.rolling_mean('grad_kurtosis'):.2f})"
                        ),
                        details={
                            "signal": "kurtosis",
                            "z_score": float(z),
                            "value": float(kurtosis_vals[-1]),
                            "rolling_mean": acc.rolling_mean("grad_kurtosis"),
                        },
                    ))

            # Check concentration ratio
            conc_vals = acc.get("grad_concentration")
            if hasattr(conc_vals, '__len__') and len(conc_vals) > 0:
                latest = float(conc_vals[-1])
                if latest > self._concentration_threshold:
                    alerts.append(Alert(
                        step=step,
                        detector=self.name,
                        severity="warning",
                        table=table_name,
                        message=(
                            f"Gradient concentration ratio {latest:.1f}x "
                            f"exceeds threshold {self._concentration_threshold}"
                        ),
                        details={
                            "signal": "concentration",
                            "concentration_ratio": latest,
                            "threshold": self._concentration_threshold,
                        },
                    ))

        return alerts
