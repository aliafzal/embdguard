"""Tests for StatAccumulator ring buffer."""
import numpy as np
from stats import StatAccumulator


def test_push_and_get():
    acc = StatAccumulator(window_size=5)
    acc.push({"grad_norm": 1.0, "n_accessed": 10})
    acc.push({"grad_norm": 2.0, "n_accessed": 20})
    assert np.array_equal(acc.get("grad_norm"), [1.0, 2.0])
    assert np.array_equal(acc.get("n_accessed"), [10.0, 20.0])


def test_window_overflow():
    acc = StatAccumulator(window_size=3)
    for i in range(5):
        acc.push({"val": float(i)})
    assert np.array_equal(acc.get("val"), [2.0, 3.0, 4.0])


def test_rolling_mean():
    acc = StatAccumulator(window_size=10)
    for i in range(4):
        acc.push({"x": float(i)})
    assert acc.rolling_mean("x") == pytest.approx(1.5)
    assert acc.rolling_mean("x", n=2) == pytest.approx(2.5)


def test_rolling_std():
    acc = StatAccumulator(window_size=10)
    for v in [2.0, 4.0, 4.0, 4.0, 5.0, 5.0, 7.0, 9.0]:
        acc.push({"x": v})
    assert acc.rolling_std("x") == pytest.approx(np.std([2, 4, 4, 4, 5, 5, 7, 9]), abs=1e-6)


def test_z_score():
    acc = StatAccumulator(window_size=10)
    for v in [1.0, 1.0, 1.0, 1.0, 1.0]:
        acc.push({"x": v})
    # All same -> std=0 -> z_score should be 0
    assert acc.z_score("x") == 0.0

    # Values with slight variance, then a spike
    acc2 = StatAccumulator(window_size=100)
    for v in [1.0, 1.1, 0.9, 1.0, 1.1, 0.9, 1.0, 1.1, 0.9, 1.0]:
        acc2.push({"x": v})
    acc2.push({"x": 10.0})
    z = acc2.z_score("x")
    assert z > 3.0  # 10 is far from mean ~1.0


def test_empty_get():
    acc = StatAccumulator()
    assert len(acc.get("nonexistent")) == 0


def test_len():
    acc = StatAccumulator()
    assert len(acc) == 0
    acc.push({"a": 1.0})
    acc.push({"a": 2.0})
    assert len(acc) == 2


def test_stat_names():
    acc = StatAccumulator()
    acc.push({"grad_norm": 1.0, "grad_max": 0.5})
    assert set(acc.stat_names) == {"grad_norm", "grad_max"}


import pytest  # noqa: E402 (needed for pytest.approx)
