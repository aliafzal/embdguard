"""Tests for anomaly detectors."""
import pytest
from embdguard.stats import StatAccumulator
from embdguard.detectors.gradient_anomaly import GradientAnomalyDetector
from embdguard.detectors.access_frequency import AccessFrequencyDetector


class TestGradientAnomalyDetector:
    def _make_stats(self, values):
        """Create a StatAccumulator with pre-loaded grad_norm values."""
        acc = StatAccumulator(window_size=100)
        for v in values:
            acc.push({"grad_norm": v})
        return {"test_table": acc}

    def test_no_alert_normal_gradients(self):
        # Steady gradient norms
        stats = self._make_stats([1.0] * 25)
        det = GradientAnomalyDetector(threshold_z=3.0, min_steps=20)
        alerts = det.check(step=25, table_stats=stats, model=None)
        assert len(alerts) == 0

    def test_alert_on_spike(self):
        # Normal values with slight variance, then a spike
        import random
        random.seed(42)
        values = [1.0 + random.uniform(-0.1, 0.1) for _ in range(25)] + [100.0]
        stats = self._make_stats(values)
        det = GradientAnomalyDetector(threshold_z=3.0, min_steps=20)
        alerts = det.check(step=26, table_stats=stats, model=None)
        assert len(alerts) == 1
        assert alerts[0].detector == "gradient_anomaly"
        assert alerts[0].table == "test_table"
        assert alerts[0].severity == "warning"
        assert alerts[0].details["z_score"] > 3.0

    def test_skips_before_min_steps(self):
        values = [1.0] * 5 + [100.0]
        stats = self._make_stats(values)
        det = GradientAnomalyDetector(threshold_z=3.0, min_steps=20)
        alerts = det.check(step=6, table_stats=stats, model=None)
        assert len(alerts) == 0  # only 6 steps < min_steps=20

    def test_table_filter(self):
        values = [1.0] * 25 + [100.0]
        stats = self._make_stats(values)
        det = GradientAnomalyDetector(tables=["other_table"])
        alerts = det.check(step=26, table_stats=stats, model=None)
        assert len(alerts) == 0  # test_table filtered out


class TestAccessFrequencyDetector:
    def _make_stats(self, id_sequences):
        """Create stats with accessed_ids pushed as lists."""
        acc = StatAccumulator(window_size=100)
        for ids in id_sequences:
            # Push accessed_ids as a list (like hooks.py does via guard.py)
            acc.push({"accessed_ids": ids})
        return {"test_table": acc}

    def test_no_alert_uniform_access(self):
        # Each item accessed once per step, evenly distributed
        id_seqs = [[i] for i in range(15)]
        stats = self._make_stats(id_seqs)
        det = AccessFrequencyDetector(concentration_threshold=5.0, min_steps=10)
        for step in range(15):
            alerts = det.check(step=step, table_stats=stats, model=None)
        # All items accessed once, ratio = 1.0 < 5.0
        assert len(alerts) == 0

    def test_alert_on_concentrated_access(self):
        # Simulate step-by-step: item 42 accessed every step, others only once
        det = AccessFrequencyDetector(concentration_threshold=3.0, min_steps=10)
        alerts = []
        for i in range(15):
            acc = StatAccumulator(window_size=100)
            acc.push({"accessed_ids": [42, 100 + i]})  # 42 every step + unique
            stats = {"test_table": acc}
            alerts = det.check(step=i, table_stats=stats, model=None)
        assert len(alerts) == 1
        assert alerts[0].detector == "access_frequency"
        assert alerts[0].details["hottest_row"] == 42

    def test_skips_before_min_steps(self):
        id_seqs = [[42]] * 5
        stats = self._make_stats(id_seqs)
        det = AccessFrequencyDetector(min_steps=10)
        alerts = det.check(step=5, table_stats=stats, model=None)
        assert len(alerts) == 0
