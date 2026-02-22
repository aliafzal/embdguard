"""Tests for the evaluation framework."""
import pytest
from src.evaluation.harness import EvalRun, EvalResult, DataConfig, AttackConfig
from src.evaluation.sensitivity import sweep, format_sweep_results
from src.evaluation.compare import compare, format_comparison
from src.detectors.access_frequency import AccessFrequencyDetector
from src.detectors.gradient_anomaly import GradientAnomalyDetector


class TestEvalResult:
    def test_precision_recall_f1(self):
        r = EvalResult(true_positives=8, false_positives=2, total_attack_steps=25)
        assert r.precision == 0.8
        assert r.recall == 1.0
        assert r.total_alerts == 10

    def test_zero_alerts(self):
        r = EvalResult(true_positives=0, false_positives=0, total_attack_steps=25)
        assert r.precision == 0.0
        assert r.recall == 0.0
        assert r.f1 == 0.0

    def test_summary_dict(self):
        r = EvalResult(true_positives=5, false_positives=1, detection_latency=8)
        d = r.summary_dict()
        assert "precision" in d
        assert "detection_latency" in d
        assert d["detection_latency"] == 8


class TestEvalRun:
    def test_basic_run(self):
        """EvalRun completes without errors and detects the attack."""
        det = AccessFrequencyDetector(concentration_threshold=5.0, min_steps=10)
        run = EvalRun(
            detectors=[det],
            data_config=DataConfig(n_users=200, n_items=500, batch_size=64),
            attack_config=AttackConfig(clean_steps=15, attack_steps=15),
        )
        result = run.execute()
        assert isinstance(result, EvalResult)
        # AccessFrequencyDetector should fire during attack phase
        assert result.true_positives > 0
        assert result.detection_latency is not None

    def test_no_false_positives_on_short_clean(self):
        """With min_steps=20, no alerts during 10 clean steps."""
        det = GradientAnomalyDetector(threshold_z=3.0, min_steps=20)
        run = EvalRun(
            detectors=[det],
            data_config=DataConfig(n_users=200, n_items=500, batch_size=64),
            attack_config=AttackConfig(clean_steps=10, attack_steps=10),
        )
        result = run.execute()
        assert result.false_positives == 0

    def test_deterministic_with_seed(self):
        """Same seed produces same results."""
        det1 = AccessFrequencyDetector(concentration_threshold=5.0, min_steps=10)
        det2 = AccessFrequencyDetector(concentration_threshold=5.0, min_steps=10)
        run1 = EvalRun(detectors=[det1], seed=123,
                       data_config=DataConfig(n_users=200, n_items=500, batch_size=64),
                       attack_config=AttackConfig(clean_steps=15, attack_steps=15))
        run2 = EvalRun(detectors=[det2], seed=123,
                       data_config=DataConfig(n_users=200, n_items=500, batch_size=64),
                       attack_config=AttackConfig(clean_steps=15, attack_steps=15))
        r1 = run1.execute()
        r2 = run2.execute()
        assert r1.true_positives == r2.true_positives
        assert r1.false_positives == r2.false_positives


class TestSweep:
    def test_sweep_runs(self):
        results = sweep(
            AccessFrequencyDetector,
            param_grid={"concentration_threshold": [3.0, 5.0], "min_steps": [10]},
            data_config=DataConfig(n_users=200, n_items=500, batch_size=64),
            attack_config=AttackConfig(clean_steps=15, attack_steps=15),
        )
        assert len(results) == 2
        assert "precision" in results[0]
        assert "concentration_threshold" in results[0]

    def test_format_sweep_results(self):
        results = [
            {"threshold": 3.0, "precision": 0.8, "recall": 1.0},
            {"threshold": 5.0, "precision": 0.9, "recall": 1.0},
        ]
        text = format_sweep_results(results)
        assert "threshold" in text
        assert "precision" in text


class TestCompare:
    def test_compare_runs(self):
        configs = [
            {"name": "access_freq", "detectors": [AccessFrequencyDetector(min_steps=10)]},
            {"name": "grad_anomaly", "detectors": [GradientAnomalyDetector(min_steps=10)]},
        ]
        results = compare(
            configs,
            data_config=DataConfig(n_users=200, n_items=500, batch_size=64),
            attack_config=AttackConfig(clean_steps=15, attack_steps=15),
        )
        assert len(results) == 2
        assert results[0]["config"] == "access_freq"
        assert results[1]["config"] == "grad_anomaly"

    def test_format_comparison(self):
        results = [
            {"config": "a", "precision": 0.8},
            {"config": "b", "precision": 0.9},
        ]
        text = format_comparison(results)
        assert "config" in text
