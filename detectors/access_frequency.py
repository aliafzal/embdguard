"""Access frequency detector for embedding tables.

Detects abnormal concentration of embedding lookups. DLAttack injects
fake users who all interact with the target item, causing its access
frequency to spike relative to organic popularity.
"""
import collections
import numpy as np

from alerts import Alert
from detectors import BaseDetector
from stats import StatAccumulator


class AccessFrequencyDetector(BaseDetector):
    """Alerts when a small set of embedding rows are accessed disproportionately."""

    def __init__(
        self,
        concentration_threshold: float = 5.0,
        min_steps: int = 10,
        tables: list[str] | None = None,
    ):
        self._concentration_threshold = concentration_threshold
        self._min_steps = min_steps
        self._tables = tables
        self._access_counts: dict[str, collections.Counter] = {}
        self._step_count = 0

    @property
    def name(self) -> str:
        return "access_frequency"

    def check(self, step, table_stats, model) -> list[Alert]:
        self._step_count += 1
        alerts = []

        for table_name, acc in table_stats.items():
            if self._tables and table_name not in self._tables:
                continue

            # Update access counts from the latest step's accessed_ids
            ids_arr = acc.get("accessed_ids")
            if len(ids_arr) == 0:
                continue
            latest_ids = ids_arr[-1]
            if not isinstance(latest_ids, (list, np.ndarray)):
                continue

            if table_name not in self._access_counts:
                self._access_counts[table_name] = collections.Counter()
            self._access_counts[table_name].update(latest_ids)

            if self._step_count < self._min_steps:
                continue

            counts = self._access_counts[table_name]
            if not counts:
                continue

            values = np.array(list(counts.values()), dtype=np.float64)
            mean_count = values.mean()
            if mean_count < 1e-10:
                continue
            max_count = values.max()
            ratio = max_count / mean_count

            if ratio > self._concentration_threshold:
                hottest_id = counts.most_common(1)[0][0]
                alerts.append(Alert(
                    step=step,
                    detector=self.name,
                    severity="warning",
                    table=table_name,
                    message=(f"Row {hottest_id} accessed {ratio:.1f}x above mean "
                             f"(count={int(max_count)}, mean={mean_count:.1f})"),
                    details={
                        "concentration_ratio": float(ratio),
                        "hottest_row": int(hottest_id),
                        "hottest_count": int(max_count),
                        "mean_count": float(mean_count),
                    },
                ))
        return alerts
