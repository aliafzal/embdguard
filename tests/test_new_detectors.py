"""Tests for new detectors: EmbeddingDrift, GradientDistribution, TemporalAccess."""
import pytest
import torch
import numpy as np
from unittest.mock import MagicMock

from src.stats import StatAccumulator
from src.detectors.embedding_drift import EmbeddingDriftDetector
from src.detectors.gradient_distribution import GradientDistributionDetector
from src.detectors.temporal_access import TemporalAccessDetector


def _mock_model(n_users=50, n_items=100, emb_dim=16):
    """Create a mock model with accessible embedding weights."""
    from dlattack_research.src.model import build_ebc, TwoTower, TwoTowerTrainTask
    ebc = build_ebc(n_users, n_items, emb_dim, device=torch.device("cpu"))
    tt = TwoTower(ebc, layer_sizes=[32, 16], device=torch.device("cpu"))
    return TwoTowerTrainTask(tt)


class TestEmbeddingDriftDetector:
    def test_no_alert_on_stable_weights(self):
        model = _mock_model()
        det = EmbeddingDriftDetector(drift_threshold_z=3.0, min_steps=5)
        # First call takes snapshot
        alerts = det.check(step=1, table_stats={}, model=model)
        assert len(alerts) == 0
        # Subsequent calls with same weights — no drift
        for i in range(2, 10):
            alerts = det.check(step=i, table_stats={}, model=model)
        assert len(alerts) == 0

    def test_alert_on_drifted_row(self):
        model = _mock_model(n_items=100, emb_dim=16)
        det = EmbeddingDriftDetector(drift_threshold_z=2.0, min_steps=3)
        # Take snapshot
        det.check(step=1, table_stats={}, model=model)
        # Massively shift one row
        with torch.no_grad():
            model.two_tower.ebc.embedding_bags["t_item_id"].weight.data[42] += 10.0
        # Run enough steps
        alerts = []
        for i in range(2, 8):
            alerts = det.check(step=i, table_stats={}, model=model)
        assert len(alerts) > 0
        drift_alert = [a for a in alerts if a.details.get("row_id") == 42]
        assert len(drift_alert) > 0
        assert drift_alert[0].detector == "embedding_drift"

    def test_skips_before_min_steps(self):
        model = _mock_model()
        det = EmbeddingDriftDetector(drift_threshold_z=2.0, min_steps=20)
        det.check(step=1, table_stats={}, model=model)
        with torch.no_grad():
            model.two_tower.ebc.embedding_bags["t_item_id"].weight.data[0] += 10.0
        alerts = det.check(step=2, table_stats={}, model=model)
        assert len(alerts) == 0  # step_count=2 < min_steps=20

    def test_table_filter(self):
        model = _mock_model()
        det = EmbeddingDriftDetector(tables=["nonexistent_table"], min_steps=3)
        det.check(step=1, table_stats={}, model=model)
        with torch.no_grad():
            model.two_tower.ebc.embedding_bags["t_item_id"].weight.data[0] += 10.0
        for i in range(2, 8):
            alerts = det.check(step=i, table_stats={}, model=model)
        assert len(alerts) == 0


class TestGradientDistributionDetector:
    def _make_stats(self, kurtosis_vals=None, concentration_vals=None):
        acc = StatAccumulator(window_size=100)
        n = max(len(kurtosis_vals or []), len(concentration_vals or []))
        for i in range(n):
            data = {}
            if kurtosis_vals and i < len(kurtosis_vals):
                data["grad_kurtosis"] = kurtosis_vals[i]
            if concentration_vals and i < len(concentration_vals):
                data["grad_concentration"] = concentration_vals[i]
            acc.push(data)
        return {"test_table": acc}

    def test_no_alert_stable_distribution(self):
        # Stable kurtosis around 0, low concentration
        stats = self._make_stats(
            kurtosis_vals=[0.5] * 25,
            concentration_vals=[2.0] * 25,
        )
        det = GradientDistributionDetector(kurtosis_z=3.0, concentration_threshold=10.0, min_steps=20)
        alerts = det.check(step=25, table_stats=stats, model=None)
        assert len(alerts) == 0

    def test_alert_on_kurtosis_spike(self):
        import random
        random.seed(42)
        kurtosis_vals = [0.5 + random.uniform(-0.1, 0.1) for _ in range(25)] + [50.0]
        stats = self._make_stats(kurtosis_vals=kurtosis_vals, concentration_vals=[2.0] * 26)
        det = GradientDistributionDetector(kurtosis_z=3.0, concentration_threshold=100.0, min_steps=20)
        alerts = det.check(step=26, table_stats=stats, model=None)
        kurtosis_alerts = [a for a in alerts if a.details.get("signal") == "kurtosis"]
        assert len(kurtosis_alerts) == 1

    def test_alert_on_high_concentration(self):
        stats = self._make_stats(
            kurtosis_vals=[0.5] * 25,
            concentration_vals=[2.0] * 24 + [15.0],
        )
        det = GradientDistributionDetector(concentration_threshold=10.0, min_steps=20)
        alerts = det.check(step=25, table_stats=stats, model=None)
        conc_alerts = [a for a in alerts if a.details.get("signal") == "concentration"]
        assert len(conc_alerts) == 1
        assert conc_alerts[0].details["concentration_ratio"] == 15.0

    def test_skips_before_min_steps(self):
        stats = self._make_stats(kurtosis_vals=[0.5] * 5 + [50.0])
        det = GradientDistributionDetector(min_steps=20)
        alerts = det.check(step=6, table_stats=stats, model=None)
        assert len(alerts) == 0


class TestTemporalAccessDetector:
    def _make_stats(self, id_sequences):
        acc = StatAccumulator(window_size=100)
        for ids in id_sequences:
            acc.push({"accessed_ids": ids})
        return {"test_table": acc}

    def test_no_alert_varied_access(self):
        # Each step accesses completely different IDs — no overlap
        det = TemporalAccessDetector(
            jaccard_threshold=0.5, burst_threshold=0.8, min_steps=10
        )
        all_alerts = []
        for i in range(15):
            acc = StatAccumulator(window_size=100)
            acc.push({"accessed_ids": list(range(i * 10, i * 10 + 5))})
            stats = {"test_table": acc}
            alerts = det.check(step=i, table_stats=stats, model=None)
            all_alerts.extend(alerts)
        assert len(all_alerts) == 0

    def test_burst_alert_on_repeated_top_k(self):
        # Item 42 appears in every step's access list
        det = TemporalAccessDetector(
            burst_window=5, burst_threshold=0.8, top_k=3, min_steps=5
        )
        all_alerts = []
        for i in range(12):
            acc = StatAccumulator(window_size=100)
            # 42 appears every step, others vary
            acc.push({"accessed_ids": [42, 42, 42, 100 + i, 200 + i]})
            stats = {"test_table": acc}
            alerts = det.check(step=i, table_stats=stats, model=None)
            all_alerts.extend(alerts)
        burst_alerts = [a for a in all_alerts if a.details.get("signal") == "burst"]
        assert len(burst_alerts) > 0
        assert any(a.details["row_id"] == 42 for a in burst_alerts)

    def test_jaccard_alert_on_high_overlap(self):
        # Same set of IDs every step
        det = TemporalAccessDetector(
            jaccard_threshold=0.3, min_steps=5
        )
        all_alerts = []
        for i in range(12):
            acc = StatAccumulator(window_size=100)
            acc.push({"accessed_ids": [1, 2, 3, 4, 5]})
            stats = {"test_table": acc}
            alerts = det.check(step=i, table_stats=stats, model=None)
            all_alerts.extend(alerts)
        jaccard_alerts = [a for a in all_alerts if a.details.get("signal") == "jaccard"]
        assert len(jaccard_alerts) > 0
        assert jaccard_alerts[0].details["jaccard"] == 1.0

    def test_skips_before_min_steps(self):
        det = TemporalAccessDetector(min_steps=20)
        for i in range(5):
            acc = StatAccumulator(window_size=100)
            acc.push({"accessed_ids": [1, 2, 3]})
            stats = {"test_table": acc}
            alerts = det.check(step=i, table_stats=stats, model=None)
        assert len(alerts) == 0
