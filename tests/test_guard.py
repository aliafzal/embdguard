"""End-to-end tests for EmbdGuard orchestrator."""
import os
import json
import tempfile
import torch
import pytest
from embdguard import EmbdGuard, GradientAnomalyDetector, AccessFrequencyDetector


def test_guard_attaches_to_model(small_model):
    guard = EmbdGuard(small_model)
    assert len(guard._hooks_list) > 0
    assert len(guard._table_stats) > 0
    guard.detach()


def test_guard_rejects_model_without_ebc():
    model = torch.nn.Linear(10, 5)
    with pytest.raises(ValueError, match="No EmbeddingBagCollection"):
        EmbdGuard(model)


def test_step_collects_stats(small_model, sample_kjt, sample_labels):
    guard = EmbdGuard(small_model)

    loss, _ = small_model(sample_kjt, sample_labels)
    loss.backward()
    alerts = guard.step()

    assert guard.step_count == 1
    # Check stats were collected for at least one table
    for tname, acc in guard._table_stats.items():
        if len(acc.get("grad_norm")) > 0:
            assert acc.get("grad_norm")[-1] > 0

    guard.detach()


def test_detector_integration(small_model, sample_kjt, sample_labels):
    """Run enough steps with a low min_steps detector to trigger checks."""
    guard = EmbdGuard(small_model)
    guard.add_detector(GradientAnomalyDetector(min_steps=3, threshold_z=100.0))

    for _ in range(5):
        loss, _ = small_model(sample_kjt, sample_labels)
        loss.backward()
        alerts = guard.step()

    # With threshold_z=100.0, no alerts expected on normal training
    assert isinstance(alerts, list)
    guard.detach()


def test_add_detector_chaining(small_model):
    guard = EmbdGuard(small_model)
    result = guard.add_detector(GradientAnomalyDetector())
    assert result is guard  # returns self for chaining
    guard.detach()


def test_summary(small_model, sample_kjt, sample_labels):
    guard = EmbdGuard(small_model)
    for _ in range(3):
        loss, _ = small_model(sample_kjt, sample_labels)
        loss.backward()
        guard.step()

    summary = guard.summary()
    assert isinstance(summary, dict)
    for tname, info in summary.items():
        assert "steps" in info
        assert "stats" in info

    guard.detach()


def test_context_manager(small_model, sample_kjt, sample_labels):
    with EmbdGuard(small_model) as guard:
        loss, _ = small_model(sample_kjt, sample_labels)
        loss.backward()
        guard.step()
    # After exit, hooks should be detached
    assert len(guard._hooks_list) == 0


def test_logging(small_model, sample_kjt, sample_labels):
    with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
        log_path = f.name

    try:
        guard = EmbdGuard(small_model, log_path=log_path)
        loss, _ = small_model(sample_kjt, sample_labels)
        loss.backward()
        guard.step()
        guard.detach()

        with open(log_path) as f:
            lines = [json.loads(line) for line in f if line.strip()]
        assert len(lines) > 0
        assert lines[0]["type"] == "stats"
    finally:
        os.unlink(log_path)


def test_get_stats(small_model, sample_kjt, sample_labels):
    guard = EmbdGuard(small_model)
    loss, _ = small_model(sample_kjt, sample_labels)
    loss.backward()
    guard.step()

    # Should be able to get stats for known table names
    for tname in guard._table_stats:
        acc = guard.get_stats(tname)
        assert acc is guard._table_stats[tname]

    guard.detach()


def test_check_interval(small_model, sample_kjt, sample_labels):
    """Detectors should only run every check_interval steps."""
    guard = EmbdGuard(small_model, check_interval=3)
    det = GradientAnomalyDetector(min_steps=1, threshold_z=0.001)
    guard.add_detector(det)

    all_alerts = []
    for i in range(6):
        loss, _ = small_model(sample_kjt, sample_labels)
        loss.backward()
        alerts = guard.step()
        if alerts:
            all_alerts.append(guard.step_count)

    # Detectors only checked at steps 3, 6 (every check_interval=3)
    for step in all_alerts:
        assert step % 3 == 0

    guard.detach()
