"""Temporal access pattern detector.

Analyzes temporal patterns in accessed row IDs across consecutive steps.
Normal training has low step-to-step overlap; poisoning produces high Jaccard
similarity (same target item appears every batch) and bursty repetition.
"""
import collections
import numpy as np

from src.alerts import Alert
from src.detectors import BaseDetector
from src.stats import StatAccumulator


class TemporalAccessDetector(BaseDetector):
    """Alerts on suspicious temporal repetition of embedding row accesses."""

    def __init__(
        self,
        jaccard_threshold: float = 0.5,
        burst_window: int = 5,
        burst_threshold: float = 0.8,
        top_k: int = 5,
        min_steps: int = 10,
        tables: list[str] | None = None,
    ):
        self._jaccard_threshold = jaccard_threshold
        self._burst_window = burst_window
        self._burst_threshold = burst_threshold
        self._top_k = top_k
        self._min_steps = min_steps
        self._tables = tables
        # Per-table history of accessed ID sets and top-K counts
        self._prev_ids: dict[str, set] = {}
        self._top_k_history: dict[str, collections.deque] = {}
        self._step_count = 0

    @property
    def name(self) -> str:
        return "temporal_access"

    def check(self, step, table_stats, model) -> list[Alert]:
        self._step_count += 1
        alerts = []

        for table_name, acc in table_stats.items():
            if self._tables and table_name not in self._tables:
                continue

            ids_arr = acc.get("accessed_ids")
            if len(ids_arr) == 0:
                continue
            latest_ids = ids_arr[-1]
            if not isinstance(latest_ids, (list, np.ndarray)):
                continue

            current_set = set(int(x) for x in latest_ids)

            if table_name not in self._top_k_history:
                self._top_k_history[table_name] = collections.deque(
                    maxlen=self._burst_window
                )

            # Count occurrences to find top-K
            counts = collections.Counter(int(x) for x in latest_ids)
            top_k_ids = set(x for x, _ in counts.most_common(self._top_k))
            self._top_k_history[table_name].append(top_k_ids)

            if self._step_count < self._min_steps:
                self._prev_ids[table_name] = current_set
                continue

            # Jaccard similarity with previous step
            if table_name in self._prev_ids:
                prev_set = self._prev_ids[table_name]
                if prev_set and current_set:
                    intersection = len(prev_set & current_set)
                    union = len(prev_set | current_set)
                    jaccard = intersection / union if union > 0 else 0.0

                    if jaccard > self._jaccard_threshold:
                        alerts.append(Alert(
                            step=step,
                            detector=self.name,
                            severity="info",
                            table=table_name,
                            message=(
                                f"High step-to-step access overlap: "
                                f"Jaccard={jaccard:.2f} (threshold={self._jaccard_threshold})"
                            ),
                            details={
                                "signal": "jaccard",
                                "jaccard": float(jaccard),
                                "intersection_size": intersection,
                                "union_size": union,
                            },
                        ))

            # Burst detection: find IDs appearing in top-K across many recent steps
            history = self._top_k_history[table_name]
            if len(history) >= self._burst_window:
                all_ids = set()
                for s in history:
                    all_ids |= s
                for rid in all_ids:
                    appearances = sum(1 for s in history if rid in s)
                    burst_score = appearances / len(history)
                    if burst_score >= self._burst_threshold:
                        alerts.append(Alert(
                            step=step,
                            detector=self.name,
                            severity="warning",
                            table=table_name,
                            message=(
                                f"Row {rid} in top-{self._top_k} accessed for "
                                f"{appearances}/{len(history)} consecutive steps "
                                f"(burst={burst_score:.2f})"
                            ),
                            details={
                                "signal": "burst",
                                "row_id": int(rid),
                                "burst_score": float(burst_score),
                                "appearances": appearances,
                                "window": len(history),
                            },
                        ))

            self._prev_ids[table_name] = current_set

        return alerts
