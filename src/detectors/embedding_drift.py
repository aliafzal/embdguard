"""Embedding drift detector.

Tracks per-row weight drift from a reference snapshot. Catches slow-and-steady
poisoning attacks that subtly shift target item embeddings over many steps —
the access frequency detector won't catch these, but cumulative drift will.
"""
import torch
import torch.nn as nn
import numpy as np

from src.alerts import Alert
from src.detectors import BaseDetector
from src.stats import StatAccumulator


class EmbeddingDriftDetector(BaseDetector):
    """Alerts when embedding rows drift anomalously far from a reference snapshot."""

    def __init__(
        self,
        drift_threshold_z: float = 3.0,
        snapshot_interval: int = 50,
        min_steps: int = 10,
        tables: list[str] | None = None,
    ):
        self._drift_threshold_z = drift_threshold_z
        self._snapshot_interval = snapshot_interval
        self._min_steps = min_steps
        self._tables = tables
        self._snapshots: dict[str, torch.Tensor] = {}
        self._step_count = 0

    @property
    def name(self) -> str:
        return "embedding_drift"

    def check(self, step, table_stats, model) -> list[Alert]:
        self._step_count += 1
        two_tower = model.two_tower if hasattr(model, "two_tower") else model
        if not hasattr(two_tower, "ebc"):
            return []

        alerts = []
        with torch.no_grad():
            for tname in two_tower.ebc.embedding_bags:
                if self._tables and tname not in self._tables:
                    continue

                weight = two_tower.ebc.embedding_bags[tname].weight.data

                # Take initial snapshot or refresh periodically
                if tname not in self._snapshots:
                    self._snapshots[tname] = weight.clone()
                    continue

                if self._step_count < self._min_steps:
                    continue

                # Compute per-row L2 drift
                ref = self._snapshots[tname]
                if ref.shape != weight.shape:
                    self._snapshots[tname] = weight.clone()
                    continue

                drifts = (weight - ref).norm(dim=1).cpu().numpy()
                mean_drift = drifts.mean()
                std_drift = drifts.std()

                if std_drift < 1e-10:
                    continue

                z_scores = (drifts - mean_drift) / std_drift
                outlier_mask = z_scores > self._drift_threshold_z
                outlier_rows = np.where(outlier_mask)[0]

                for row in outlier_rows:
                    alerts.append(Alert(
                        step=step,
                        detector=self.name,
                        severity="warning",
                        table=tname,
                        message=(
                            f"Row {row} drifted {z_scores[row]:.1f}\u03c3 from reference "
                            f"(L2={drifts[row]:.4f}, mean={mean_drift:.4f})"
                        ),
                        details={
                            "row_id": int(row),
                            "z_score": float(z_scores[row]),
                            "drift_l2": float(drifts[row]),
                            "mean_drift": float(mean_drift),
                            "std_drift": float(std_drift),
                        },
                    ))

                # Refresh snapshot periodically
                if self._step_count % self._snapshot_interval == 0:
                    self._snapshots[tname] = weight.clone()

        return alerts
